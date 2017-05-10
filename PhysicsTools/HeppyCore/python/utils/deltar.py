# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import math
import copy
import PhysicsTools.HeppyCore.configuration

DEFAULT_DRMAX = 0.3
DEFAULT_DRMIN = 1e-5

def deltaR2( e1, p1, e2=None, p2=None):
    """Take either 4 arguments (eta,phi, eta,phi) or two particles that have 'eta', 'phi' methods)"""
    if (e2 == None and p2 == None):
        if PhysicsTools.HeppyCore.configuration.Collider.BEAMS == 'ee':
            return deltaR2(e1.theta(),e1.phi(), p1.theta(), p1.phi())
        else:
            return deltaR2(e1.eta(),e1.phi(), p1.eta(), p1.phi())
    de = e1 - e2
    dp = deltaPhi(p1, p2)
    return de*de + dp*dp


def deltaR( *args ):
    return math.sqrt( deltaR2(*args) )


def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = p1 - p2
    while res > math.pi:
        res -= 2*math.pi
    while res < -math.pi:
        res += 2*math.pi
    return res


def inConeCollection(pivot, particles,
                     deltaRMax = DEFAULT_DRMAX,
                     deltaRMin = DEFAULT_DRMIN):
    '''Returns the list of particles that are less than deltaRMax away from pivot.'''
    dR2Max = deltaRMax ** 2
    dR2Min = deltaRMin ** 2
    results = []
    for ptc in particles:
        dR2 = deltaR2(pivot, ptc)
        if dR2Min <= dR2 < dR2Max:
            results.append(ptc)
    return results


def cleanObjectCollection(ptcs, masks, deltaRMax=DEFAULT_DRMAX):
    '''returns a tuple clean_ptcs, dirty_ptcs,
    where:
    - dirty_ptcs is the list of particles in ptcs that are matched to a particle
    in masks.
    - clean_ptcs is the list of particles in ptcs that are NOT matched to a
    particle in masks.
    
    The matching is done within a cone of size deltaRMin.
    '''
    if len(ptcs)==0 or len(masks)==0:
        return ptcs, []
    dR2Max = deltaRMax ** 2
    clean_ptcs = []
    dirty_ptcs = []
    for ptc in ptcs:
        ok = True 
        for mask in masks:
            dR2 = deltaR2(ptc, mask)
            if dR2 < dR2Max:
                ok = False
        if ok:
            clean_ptcs.append( ptc )
        else:
            dirty_ptcs.append( ptc )
    return clean_ptcs, dirty_ptcs


def cleanObjectCollection2(ptcs, masks, deltaRMax=DEFAULT_DRMAX):
    '''returns the list of particles in ptcs that are NOT matched to a
    particle in masks.
    
    The matching is done within a cone of size deltaRMin.
    
    The algorithm is different than in cleanObjectCollection, but the results are the same.
    Another difference with respect to cleanObjectCollection is that the list of dirty
    objects is not returned as well. 
    '''
    if len(ptcs)==0:
        return ptcs
    dR2Max = deltaRMax ** 2
    clean_ptcs = copy.copy( ptcs )
    for mask in masks:
        tooClose = []
        for idx, ptc in enumerate(clean_ptcs):
            dR2 = deltaR2(ptc, mask)
            if dR2 < dR2Max:
                tooClose.append( idx )
        nRemoved = 0
        for idx in tooClose:
            # yes, everytime an object is removed, the list of objects is updated
            # so need to update the index accordingly.
            # example: to remove : ele 1 and 2
            #  first, ele 1 is removed
            #  -> ele 2 is now at index 1
            # one should again remove the element at index 1
            idx -= nRemoved
            del clean_ptcs[idx]
            nRemoved += 1 
    return clean_ptcs


def bestMatch(ptc, matchCollection):
    '''Return the best match to ptc in matchCollection,
    which is the closest ptc in delta R,
    together with the squared distance dR2 between ptc
    and the match.'''
    deltaR2Min = float('+inf')
    bm = None
    for match in matchCollection:
        dR2 = deltaR2(ptc, match)
        if dR2 < deltaR2Min:
            deltaR2Min = dR2
            bm = match
    return bm, deltaR2Min


def matchObjectCollection(ptcs, matchCollection,
                          deltaRMax=DEFAULT_DRMAX, filter = lambda x,y : True):
    pairs = {}
    if len(ptcs)==0:
        return pairs
    if len(matchCollection)==0:
        return dict( list(zip(ptcs, [None]*len(ptcs))) )
    dR2Max = deltaRMax ** 2
    for ptc in ptcs:
        bm, dr2 = bestMatch( ptc, [mob for mob in matchCollection if filter(object,mob)] )
        if dr2 < dR2Max:
            pairs[ptc] = bm
        else:
            pairs[ptc] = None            
    return pairs


def matchObjectCollection2(ptcs, matchCollection,
                           deltaRMax=DEFAULT_DRMAX):
    '''Univoque association of an element from matchCollection to an element of ptcs.
    Returns a list of tuples [(ptc, matched_to_ptc), ...].
    particles in ptcs and matchCollection get the "matched" attribute,
    true is they are part of a matched tuple.
    By default, the matching is true only if delta R is smaller than 0.3.
    '''

    pairs = {}
    if len(ptcs)==0:
        return pairs
    if len(matchCollection)==0:
        return dict( list(zip(ptcs, [None]*len(ptcs))) )
    # build all possible combinations
    allPairs = [(deltaR2(ptc, match), (ptc, match))
                for ptc in ptcs for match in matchCollection]
    allPairs.sort ()

    # to flag already matched objects
    # FIXME this variable remains appended to the object, I do not like it
    for ptc in ptcs:
        ptc.matched = False
    for match in matchCollection:
        match.matched = False

    dR2Max = deltaRMax ** 2
    for dR2, (ptc, match) in allPairs:
        if dR2 > dR2Max:
            break
        if dR2 < dR2Max and ptc.matched == False and match.matched == False:
            ptc.matched = True
            match.matched = True
            pairs[ptc] = match

    for ptc in ptcs:
        if ptc.matched == False:
            pairs[ptc] = None

    return pairs
    # by now, the matched attribute remains in the objects, for future usage
    # one could remove it with delattr (object, attrname)


def matchObjectCollection3(ptcs, matchCollection,
                           deltaRMax=DEFAULT_DRMAX,
                           filter_func=None):
    '''Univoque association of an element from matchCollection to an element of ptcs.
    Returns a list of tuples [(ptc, matched_to_ptc), ...].
    particles in ptcs and matchCollection get the "matched" attribute,
    true is they are part of a matched tuple.
    By default, the matching is true only if delta R is smaller than 0.3.
    '''

    if filter_func is None:
        filter_func = lambda x,y : True
    pairs = {}
    if len(ptcs)==0:
        return pairs
    if len(matchCollection)==0:
        return dict( zip(ptcs, [None]*len(ptcs)) )
    # build all possible combinations

    ptc_coords = [ (o.eta(),o.phi(),o) for o in ptcs ]
    matched_coords = [ (o.eta(),o.phi(),o) for o in matchCollection ]
    allPairs = [(deltaR2 (oeta, ophi, meta, mphi), (ptc, match))
                for (oeta,ophi,ptc) in ptc_coords
                for (meta,mphi,match) in matched_coords
                if abs(oeta-meta)<=deltaRMax and filter_func(ptc,match) ]
    #allPairs = [(deltaR2 (object.eta(), object.phi(), match.eta(), match.phi()), (object, match)) for object in objects for match in matchCollection if filter(object,match) ]
    allPairs.sort ()
    #
    # to flag already matched objects
    # FIXME this variable remains appended to the object, I do not like it

    for ptc in ptcs:
        ptc.matched = False
    for match in matchCollection:
        match.matched = False
        
    dR2Max = deltaRMax ** 2
    for dR2, (ptc, match) in allPairs:
        if dR2 > dR2Max:
            break
        if dR2 < dR2Max and ptc.matched == False and match.matched == False:
            ptc.matched = True
            match.matched = True
            pairs[ptc] = match

    for ptc in ptcs:
        if ptc.matched == False:
            pairs[ptc] = None

    return pairs
    # by now, the matched attribute remains in the objects, for future usage
    # one could remove it with delattr (object, attrname)


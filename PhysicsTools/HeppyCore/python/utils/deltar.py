# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import math
import copy

def deltaR2( e1, p1, e2=None, p2=None):
    """Take either 4 arguments (eta,phi, eta,phi) or two objects that have 'eta', 'phi' methods)"""
    if (e2 == None and p2 == None):
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


def inConeCollection(pivot, particles, deltaRMax, deltaRMin=1e-5):
    '''Returns the list of particles that are less than deltaRMax away from pivot.'''
    dR2Max = deltaRMax ** 2
    dR2Min = deltaRMin ** 2
    results = []
    for ptc in particles:
        dR2 = deltaR2(pivot.eta(), pivot.phi(), ptc.eta(), ptc.phi()) 
        if dR2Min < dR2 < dR2Max:
            results.append(ptc)
    return results

def matchObjectCollection3 ( objects, matchCollection, deltaRMax = 0.3, filter = lambda x,y : True ):
    '''Univoque association of an element from matchCollection to an element of objects.
    Reco and Gen objects get the "matched" attribute, true is they are re part of a matched tulpe.
    By default, the matching is true only if delta R is smaller than 0.3. 
    '''
    #
                                                                                                                                                                                                                                       
    pairs = {}
    if len(objects)==0:
            return pairs
    if len(matchCollection)==0:
            return dict( zip(objects, [None]*len(objects)) )
    # build all possible combinations

    objectCoords = [ (o.eta(),o.phi(),o) for o in objects ]
    matchdCoords = [ (o.eta(),o.phi(),o) for o in matchCollection ]
    allPairs = [(deltaR2 (oeta, ophi, meta, mphi), (object, match)) for (oeta,ophi,object) in objectCoords for (meta,mphi,match) in matchdCoords if abs(oeta-meta)<=deltaRMax and filter(object,match) ]
    #allPairs = [(deltaR2 (object.eta(), object.phi(), match.eta(), match.phi()), (object, match)) for object in objects for match in matchCollection if filter(object,match) ]
    allPairs.sort ()
    #
    # to flag already matched objects
    # FIXME this variable remains appended to the object, I do not like it

    for object in objects:
        object.matched = False
    for match in matchCollection:
        match.matched = False
    #

    deltaR2Max = deltaRMax * deltaRMax
    for dR2, (object, match) in allPairs:
        if dR2 > deltaR2Max:
            break
        if dR2 < deltaR2Max and object.matched == False and match.matched == False:
            object.matched = True
            match.matched = True
            pairs[object] = match
    #

    for object in objects:
       if object.matched == False:
           pairs[object] = None
    #

    return pairs
    # by now, the matched attribute remains in the objects, for future usage
    # one could remove it with delattr (object, attrname)




def cleanObjectCollection2( objects, masks, deltaRMin ):
    '''Masks objects using a deltaR cut, another algorithm (same results).'''
    if len(objects)==0:
        return objects
    deltaR2Min = deltaRMin*deltaRMin
    cleanObjects = copy.copy( objects )
    for mask in masks:
        tooClose = []
        for idx, object in enumerate(cleanObjects):
            dR2 = deltaR2( object.eta(), object.phi(),
                           mask.eta(), mask.phi() )
            if dR2 < deltaR2Min:
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
            del cleanObjects[idx]
            nRemoved += 1 
    return cleanObjects


def cleanObjectCollection( objects, masks, deltaRMin ):
    '''Masks objects using a deltaR cut.'''
    if len(objects)==0 or len(masks)==0:
        return objects, []
    deltaR2Min = deltaRMin*deltaRMin
    cleanObjects = []
    dirtyObjects = []
    for object in objects:
        ok = True 
        for mask in masks:
            dR2 = deltaR2( object.eta(), object.phi(),
                           mask.eta(), mask.phi() )
            if dR2 < deltaR2Min:
                ok = False
        if ok:
            cleanObjects.append( object )
        else:
            dirtyObjects.append( object )
    return cleanObjects, dirtyObjects

def bestMatch( object, matchCollection):
    '''Return the best match to object in matchCollection, which is the closest object in delta R'''
    deltaR2Min = float('+inf')
    bm = None
    for match in matchCollection:
        dR2 = deltaR2( object.eta(), object.phi(),
                       match.eta(), match.phi() )
        if dR2 < deltaR2Min:
            deltaR2Min = dR2
            bm = match
    return bm, deltaR2Min


def matchObjectCollection( objects, matchCollection, deltaR2Max):
    pairs = {}
    if len(objects)==0:
        return pairs
    if len(matchCollection)==0:
        return dict( zip(objects, [None]*len(objects)) )
    for object in objects:
        bm, dr2 = bestMatch( object, matchCollection )
        if dr2<deltaR2Max:
            pairs[object] = bm
        else:
            pairs[object] = None            
    return pairs


def matchObjectCollection2 ( objects, matchCollection, deltaRMax = 0.3 ):
    '''Univoque association of an element from matchCollection to an element of objects.
    Reco and Gen objects get the "matched" attribute, true is they are re part of a matched tulpe.
    By default, the matching is true only if delta R is smaller than 0.3.
    '''
    
    pairs = {}
    if len(objects)==0:
            return pairs
    if len(matchCollection)==0:
            return dict( zip(objects, [None]*len(objects)) )
    # build all possible combinations
    allPairs = [(deltaR2 (object.eta(), object.phi(), match.eta(), match.phi()), (object, match)) for object in objects for match in matchCollection]
    allPairs.sort ()

    # to flag already matched objects
    # FIXME this variable remains appended to the object, I do not like it
    for object in objects:
        object.matched = False
    for match in matchCollection:
        match.matched = False
    
    deltaR2Max = deltaRMax * deltaRMax
    for dR2, (object, match) in allPairs:
	if dR2 > deltaR2Max:
		break
        if dR2 < deltaR2Max and object.matched == False and match.matched == False:
            object.matched = True
            match.matched = True
            pairs[object] = match
    
    for object in objects:
       if object.matched == False:
	   pairs[object] = None

    return pairs
    # by now, the matched attribute remains in the objects, for future usage
    # one could remove it with delattr (object, attrname)




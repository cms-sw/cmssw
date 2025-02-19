# Benedikt Hegner, DESY
# benedikt.hegner@cern.ch
#
# this tool is based on Luca Lista's tree drawer module


class ParticleDecayDrawer(object):
    """Draws particle decay tree """
    
    def __init__(self):
        print "Init particleDecayDrawer"
        #  booleans: printP4 printPtEtaPhi printVertex   
        
    def _accept(self, candidate, skipList):
        if candidate in skipList: return False;
        return self._select(candidate)
        
    def _select(self, candidate):
        return candidate.status() == 3
         
    def _hasValidDaughters(self, candidate):
        nDaughters = candidate.numChildren()
        for i in xrange(nDaughters):
            if self._select(candidate.listChildren()[i]): return True
        return False        

    def _printP4(self, candidate):
        return " "

    def _decay(self, candidate, skipList):

          out = str()
          if candidate in skipList:
              return ""
          skipList.append(candidate)
    
          id = candidate.pdg_id()
          # here the part about the names :-(
          out += str(id) + self._printP4(candidate)
    
          validDau = 0
          nOfDaughters = candidate.numChildren()
          for i in xrange(nOfDaughters):
              if self._accept(candidate.listChildren()[i], skipList): validDau+=1
          if validDau == 0: return out
    
          out += " ->"
    
          for i in xrange(nOfDaughters):
              d = candidate.listChildren()[i]
              if self._accept(d, skipList):
                  decString = self._decay(d, skipList)
                  if ("->" in decString):  out += " ( %s ) " %decString
                  else:  out += " %s" %decString
          return out

    def draw(self, particles): 
        """ draw decay tree from list(HepMC.GenParticles)""" 
        skipList = []
        nodesList = []
        momsList = []  
        for particle in particles:
            if particle.numParents() > 1:
                if self._select(particle):
                    skipList.append(particle)
                    nodesList.append(particle)
                    for j in xrange(particle.numParents()):
                        mom = particle.listParents()[j]
                        while (mom.mother()):# != None ):
                            mom = mom.mother()
                        if self._select(mom):
                            momsList.append(mom)

        print "-- decay --"  
        if len(momsList) > 0:
            if len(momsList) > 1:
                for m in xrange(len(momsList)):
                    decString = self._decay( momsList[m], skipList)
                    if len(decString) > 0:
                       print "{ %s } " %decString
            else:
                print self._decay(momsList[0], skipList)   
  
        if len(nodesList) > 0:
            print "-> "
            if len(nodesList) > 1:
                for node in nodesList:
                   skipList.remove(node)
                   decString = self._decay(node, skipList)
                   if len(decString) > 0:
                       if "->" in decString:  print " ( %s ) " %decString
                       else:  print " " + decString
            else:
                skipList.remove(nodesList[0])
                print self._decay(nodesList[0], skipList)
          
        print
    

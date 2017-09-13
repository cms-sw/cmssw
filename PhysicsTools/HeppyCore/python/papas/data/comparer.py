from numpy.testing.utils import assert_allclose

class ParticlesComparer(object):
    '''  Checks that two lists of presorted particles are identical
          will stop on an assert if things are different
          Note that there is an issue with accuracy of particle mass as assessed via TLorentzVector
    '''
    def __init__(self, particlesA, particlesB):
        ''' Simple check that two sets of sensibly sorted particles are the same
            is relatively naive but sufficient so far
        '''
        self.A = particlesA
        self.B = particlesB
        #self.history = history
        
        assert(len(self.A)==len(self.B))
        
        for i in range(len(self.A)):
            
            try: 
                assert(self.A[i].pdgid()==  self.B[i].pdgid() )
                assert_allclose(self.A[i].p4().Y(), self.B[i].p4().Y(), rtol=1e-8, atol=0.0000001 )
                assert_allclose(self.A[i].p4().Z(), self.B[i].p4().Z(), rtol=1e-8, atol=0.0000001 ) 
                assert_allclose(self.A[i].q(),  self.B[i].q(),  rtol=1e-8, atol=0.0000001 ) 
                assert_allclose(self.A[i].p4().X(), self.B[i].p4().X(), rtol=1e-8, atol=0.0000001 )
                assert_allclose(self.A[i].eta(),self.B[i].eta(), rtol=1e-8, atol=0.0000001 )
                assert_allclose(self.A[i].phi(),self.B[i].phi(), rtol=1e-8, atol=0.0000001 )
                assert_allclose(self.A[i].theta(),self.B[i].theta(), rtol=1e-8, atol=0.0000001 )
                assert_allclose(self.A[i].pt(),self.B[i].pt(), rtol=1e-8, atol=0.0000001 )
                assert_allclose(self.A[i].e(),self.B[i].e(), rtol=1e-8, atol=0.0000001 )
                assert_allclose(self.A[i].p4().M(), self.B[i].p4().M(), rtol=1e-5, atol=0.00001 )  #reduced accuracy becasue of root issue  
                
                      
            except AssertionError:
                print i
                #print self.history.summary_of_links(self.A[i].uniqueid) ,self.B[i]  
                print self.A[i].p4().X(), self.B[i].p4().X()
                print self.A[i].p4().Y(), self.B[i].p4().Y()
                print self.A[i].p4().Z(), self.B[i].p4().Z()
                print self.A[i].p4().e(),self.B[i].p4().e()
                print self.A[i].pt().pt(),self.B[i].p4().pt()
                print self.A[i].p4().M(),self.B[i].p4().M()
                assert(False)
          
        
class ClusterComparer(object):
    '''  Checks that two dicts of clusters are identical. Will 
         stop with an assert if differences are found
    '''

    def __init__(self, clustersA, clustersB):
        ''' Simple check that two dicts of clusters are the same
        '''
        self.A = clustersA
        self.B = clustersB
        self.A = sorted( self.A.values(),
                            key = lambda ptc: ptc.energy, reverse = True)        
        self.B = sorted( self.B.values(),
                            key = lambda ptc: ptc.energy, reverse = True)         
        assert(len(self.A)==len(self.B))
        
        for i in range(len(self.A)):
            #print self.A[i]
            #print self.B[i]
            AS = sorted( self.A[i].subclusters, key = lambda x: x.uniqueid)
            BS = sorted( self.B[i].subclusters, key = lambda x: x.uniqueid)            
            for j in range(len(self.A[i].subclusters)): 
                assert (AS[j].uniqueid==BS[j].uniqueid)
            
            # really ought to be checked for non merged clusters
            assert_allclose(self.A[i].energy, self.B[i].energy, rtol  = 1e-12, atol=0.00000000001 )
            #angular size does not make sense for merged clusters
            #and should not be used
            #assert_allclose(self.A[i].angular_size(),  self.B[i].angular_size(), rtol  = 1e-12, atol=0.00000000001 )
            assert_allclose(self.A[i].position.Theta(), self.B[i].position.Theta(), rtol = 1e-12, atol = 0.00000000001 )
            assert_allclose(self.A[i].position.Phi(), self.B[i].position.Phi(), rtol = 1e-12, atol = 0.00000000001 )
            assert_allclose(self.A[i].position.Mag(), self.B[i].position.Mag(), rtol = 1e-12, atol = 0.00000000001 )
            assert_allclose(self.A[i].position.X(), self.B[i].position.X(), rtol = 1e-12, atol = 0.00000000001 )
            assert_allclose(self.A[i].position.Y(), self.B[i].position.Y(), rtol = 1e-12, atol = 0.00000000001 )
            assert_allclose(self.A[i].position.Z(), self.B[i].position.Z(), rtol = 1e-12, atol = 0.00000000001 ) 
            assert(len(self.A[i].subclusters)==len(self.B[i].subclusters) )
            
            # size and angularsize are not "valid" for merged clusters, however they need to be checked for other 
            # non merged clusters
            if len(self.A[i].subclusters)==1:
                assert_allclose(self.A[i]._size, self.B[i]._size, rtol  = 1e-12, atol=0.00000000001 )
                assert_allclose(self.A[i]._angularsize, self.B[i]._angularsize, rtol  = 1e-12, atol=0.00000000001 )
                           
                
                
            
class TrackComparer(object):

    def __init__(self,tracksA,tracksB):
        ''' Simple check that two dicts of tracks are the same
            will stop on an assert if things are different
            is relatively naive and may not be complete.
        '''
        self.A = tracksA
        self.B = tracksB
        self.A = sorted( self.A.values(),
                            key = lambda ptc: ptc.energy, reverse=True)        
        self.B = sorted( self.B.values(),
                            key = lambda ptc: ptc.energy, reverse=True)         
        assert(len(self.A)==len(self.B))
        
        for i in range(len(self.A)):
            assert_allclose(self.A[i].energy,  self.B[i].energy,  rtol=1e-8, atol=0.0000001 )
            assert_allclose(self.A[i].pt,      self.B[i].pt,      rtol=1e-8, atol=0.0000001 )
            assert_allclose(self.A[i].charge , self.B[i].charge , rtol=1e-8, atol=0.0000001 )
            
            
                    
            
                    
            
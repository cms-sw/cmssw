from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducer import H2TauTauTreeProducer

class H2TauTauTreeProducerTauTau( H2TauTauTreeProducer ):
  '''Tree producer for the H->tau tau analysis'''
  
  def declareVariables(self, setup):
    
    super(H2TauTauTreeProducerTauTau, self).declareVariables(setup)
    
    self.bookTau(self.tree, 'l1')
    self.bookTau(self.tree, 'l2')
    
    self.bookGenParticle(self.tree, 'l1_gen')
    self.bookGenParticle(self.tree, 'l2_gen')

    self.bookParticle(self.tree, 'l1_gen_vis')
    self.bookParticle(self.tree, 'l2_gen_vis')
    
  def process(self, event):
             
    super(H2TauTauTreeProducerTauTau, self).process(event)
    
    tau1 = event.leg1
    tau2 = event.leg2
        
    self.fillTau(self.tree, 'l1', tau1 )
    self.fillTau(self.tree, 'l2', tau2 )

    if hasattr(tau1, 'genl1') : self.fillGenParticle(self.tree, 'l1_gen', tau1.genl )
    if hasattr(tau2, 'genl2') : self.fillGenParticle(self.tree, 'l2_gen', tau2.genl )

    # RIC: hasattr has depth=1, butthis would be nicer 
    # http://code.activestate.com/recipes/577346-getattr-with-arbitrary-depth/
    # may think of putting it into some utils
    tau1po = event.leg1.physObj
    tau2po = event.leg2.physObj

    # save the p4 of the visible tau products at the generator level
    # make sure that the reco tau matches with a gen tau that decays into hadrons
    if hasattr(tau1po, 'genJet') and hasattr(tau1,'genl') and abs(tau1.genl.pdgId()) == 15 and tau1.physObj.genJet() : 
        self.fillParticle(self.tree, 'l1_gen_vis', tau1.physObj.genJet() )
    if hasattr(tau2po, 'genJet') and hasattr(tau2,'genl') and abs(tau2.genl.pdgId()) == 15 and tau2.physObj.genJet() : 
        self.fillParticle(self.tree, 'l2_gen_vis', tau2.physObj.genJet() )
      
    self.fillTree(event)

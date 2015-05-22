from CMGTools.TTHAnalysis.treeReAnalyzer import *

from ROOT import TLorentzVector, TVectorD
import ROOT

def computeMT2(visaVec, visbVec, metVec):
    
    import array
    import numpy

    from ROOT import Davismt2
    davismt2 = Davismt2()    

    metVector = array.array('d',[0.,metVec.px(), metVec.py()])
    visaVector = array.array('d',[0.,visaVec.px(), visaVec.py()])
    visbVector = array.array('d',[0.,visbVec.px(), visbVec.py()])

    davismt2.set_momenta(visaVector,visbVector,metVector);
    davismt2.set_mn(0);

    return davismt2.get_mt2()



class EventVarsMT2:
    def __init__(self):
        ROOT.gSystem.Load("libFWCoreFWLite.so")
        ROOT.gSystem.Load("libDataFormatsFWLite.so")
        ROOT.AutoLibraryLoader.enable()
#        ROOT.gSystem.Load("libCMGToolsTTHAnalysis.so")   

        self.branches = [ "deltaPhiMin4", "deltaPhiMin3", "mt2_had_30" ] 

        from ROOT import RecoilCorrector
        recoilCorr = RecoilCorrector("fileToCorrect.root",123456)    
        recoilCorr.addDataFile("fileZmumuData.root"); 
        recoilCorr.addMCFile("fileZmumuMC.root"); 

    def listBranches(self):
        return self.branches[:]
    def __call__(self,event):
        # make python lists as Collection does not support indexing in slices
        leps = [l for l in Collection(event,"lep","nlep",10)]
        jets = [j for j in Collection(event,"jet","njet",100)]
        (met, metphi)  = event.met_pt, event.met_phi
        metp4 = ROOT.reco.Particle.LorentzVector(met*cos(metphi),met*sin(metphi),0,met)

        genLepsp4 = [l for l in Collection(event,"genLep","ngenLep",10)]

        njet = len(jets); nlep = len(leps)
        # prepare output
        ret = dict([(name,0.0) for name in self.branches])

###### smear MET with Recoil Correction for V + Jets

        # fill output
        if len(genLepsp4)>=2:

            genZp4 = ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 )
            for l in genLepsp4:
                genZp4 += ROOT.reco.Particle.LorentzVector( l.p4().Px(), l.p4().Py(), l.p4().Pz(), l.p4().Energy() )
#            print 'genZ',genZp4.Pt()
         
            # those need to be added to the tree
            # ZGen_pt, ZGen_phi, ZGen_rap 
            # ZReco_pt, ZReco_phi

            (met_corr, metphi_corr)  = (met, metphi)  
#            recoilCorr.CorrectType2( met_corr, met_corr,
#                                     genZp4.Pt(), genZp4.Phi(),
#                                     ZReco_pt, ZReco_phi,
#                                     u1_dummy, u2_dummy,
#                                     RecoilCorrResolutionNSigmaU2, RecoilCorrResolutionNSigmaU1, RecoilCorrScaleNSigmaU1,
#                                     rapBin,false); 
            

        # fill output
        if njet >= 2:


###### for now clone of the code in ttHCoreEventAnalyzer.py

            deltaPhiMin_had4 = 999.
            for n,j in enumerate(jets):
                if n>3:  break
                thisDeltaPhi = abs( deltaPhi( j.phi, metphi ) )
                if thisDeltaPhi < deltaPhiMin_had4 : deltaPhiMin_had4 = thisDeltaPhi
            ret["deltaPhiMin4"] = deltaPhiMin_had4

            deltaPhiMin_had3 = 999.
            for n,j in enumerate(jets):
                if n>2:  break
                thisDeltaPhi = abs( deltaPhi( j.phi, metphi ) )
                if thisDeltaPhi < deltaPhiMin_had3 : deltaPhiMin_had3 = thisDeltaPhi
            ret["deltaPhiMin3"] = deltaPhiMin_had3

 
###### for now clone of the code in ttHTopoVarAnalyzer.py

            objects40jc = [ j for j in jets if j.pt > 40 and abs(j.eta)<2.5]
            if len(objects40jc)>=2:

                from ROOT import Hemisphere
                from ROOT import ReclusterJets
                
                from ROOT import Davismt2
                davismt2 = Davismt2()

                pseudoViaKtJet1_had = ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 )
                pseudoViaKtJet2_had = ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 )
                objects  = ROOT.std.vector(ROOT.reco.Particle.LorentzVector)()
                for jet in objects40jc:
                    jetp4 = ROOT.reco.Particle.LorentzVector(jet.p4().Px(), jet.p4().Py(), jet.p4().Pz(), jet.p4().Energy())
                    objects.push_back(jetp4)
                    
                    hemisphereViaKt = ReclusterJets(objects, 1.,50)
                    groupingViaKt=hemisphereViaKt.getGroupingExclusive(2)

                    if len(groupingViaKt)>=2:
                        pseudoViaKtJet1_had = groupingViaKt[0]
                        pseudoViaKtJet2_had = groupingViaKt[1]
                        ret["mt2_had_30"] = computeMT2(pseudoViaKtJet1_had, pseudoViaKtJet2_had, metp4)

        print 'mt2(had-original)=',event.mt2,'--- mt2ViaKt_had(original)=',event.mt2ViaKt_had,'--- mt2(new)=',ret["mt2_had_30"]

        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("mt2")
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = EventVars2LSS()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        

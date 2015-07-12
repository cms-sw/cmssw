import ROOT
ROOT.gSystem.Load("libFWCoreFWLite")
ROOT.AutoLibraryLoader.enable()


class PyJet(object):
    def __init__(self,p4):
        self.LV = ROOT.math.XYZTLorentzVector(p4.px(),p4.py(),p4.pz(),p4.energy())
        self.rawFactor = 1.0
        self.matched=0
    def setConstituents(self,constituents):
        self.constituents = constituents


    def p4(self):
        return self.LV

    def numberOfDaughters(self):
        return len(constituents)

    def daughter(self,i):
        if i<len(constituents):
            return constituents[i]
        else:
            return None

    def rawFactor(self):
        return self.rawFactor


    def jetArea(self):
            return self.area

    def __getattr__(self, name):
        return getattr(self.LV,name)



class PyJetToolbox(object):
    
    def __init__(self,collection):
        self.collection = collection
        self.p4s = ROOT.std.vector("math::XYZTLorentzVector")()
        for p in collection:
            self.p4s.push_back(p.p4())


        self.doMassDrop = False
        self.doPrunning = False
        self.doSubjets = False
        self.doSoftDrop = False
        self.doNTau = False

    def setInterface(self,doArea,ktpower,rparam,active_area_repeats=1,ghost_area = 0.01,ghost_eta_max = 5.0,rho_eta_max = 4.4):        
        if doArea:
            self.interface = ROOT.cmg.FastJetInterface(self.p4s,ktpower,rparam,active_area_repeats,ghost_area,ghost_eta_max,rho_eta_max)
        else:    
            self.interface = ROOT.cmg.FastJetInterface(self.p4s,ktpower,rparam)


    def setMassDrop(self,activate,mu=0.667 ,y=0.08):
        self.doMassDrop = activate
        self.massdrop = {'mu':mu,'y':y}


    def setSubjets(self,activate,style = 'inc',setting = 2):
        self.doSubjets = activate
        self.subjets = {'style':style,'setting':setting}


    def setPruning(self,activate,zcut = 0.1,rcutfactor = 0.5):
        self.doPrunning = activate       
        self.prunning = {'zcut':zcut,'rcutfactor':rcutfactor}

    def setSoftDrop(self,activate,beta=0.0,zcut=0.1,R0=0.4):
        self.doSoftDrop = activate
        self.softdrop = {'beta':beta,'zcut':zcut,'R0':R0}


    def setNtau(self,activate,NMAX = 4,measureDef = 0 , axesDef = 6 , beta= 1.0 , R0 = 0.8 , Rcutoff = -999.0,akAxisR0 = -999.0, nPass = -999 ):
        self.doNTau = activate
        self.ntau = {'NMAX':NMAX,'measureDef':measureDef,'axesDef':axesDef, 'beta':beta,'R0':R0,'Rcutoff':Rcutoff,'akAxesR0':-999.0, 'nPass':-999}
        

    def convert(self,lorentzVectors,isFat = False,isJet=True):
        output = []

        for LV in lorentzVectors:
            output.append(PyJet(LV))
        for i,jet in enumerate(output):
            jet.area = self.interface.getArea(isJet,i)
            jet.constituents=[]
            constituents = self.interface.getConstituents(isJet,i)
            for c in constituents:
                jet.constituents.append(self.collection[c])
            if isFat:
                if self.doPrunning:
                    self.interface.prune(isJet,self.prunning['zcut'],self.prunning['rcutfactor'])
                    jet.prunedJet = self.convert(self.interface.get(isJet),False,isJet)
                if self.doSoftDrop:
                    self.interface.softDrop(isJet,self.softdrop['beta'],self.softdrop['zcut'],self.softdrop['R0'])
                    jet.softDropJet = self.convert(self.interface.get(not isJet),False,isJet)[i]
                if self.doSubjets:
                    if self.subjets['style'] == 'inc':
                        self.interface.makeSubJets(i,self.subjets['setting'])
                        jet.subjets = self.convert(self.interface.get(True),False,False)
                    else:    
                        self.interface.makeSubJetsUpTo(i,self.subjets['setting'])
                        jet.subjets = self.convert(self.interface.get(True),False,False)
                    if self.doNTau:
                        jet.Ntau = self.interface.nSubJettiness(i,self.ntau['NMAX'],self.ntau['measureDef'],self.ntau['axesDef'],self.ntau['beta'],self.ntau['R0'],self.ntau['Rcutoff'],self.ntau['akAxesR0'],self.ntau['nPass'])
                if self.doMassDrop:
                    mu= ROOT.Double(self.massdrop['mu'])
                    y= ROOT.Double(self.massdrop['y'])
                    jet.massDropTag =  self.interface.massDropTag(i,mu,y)
                    jet.massDrop = (mu,y)
        return output
            
    def inclusiveJets(self,ptmin = 0.0,isFat=True):
        self.interface.makeInclusiveJets(ptmin)
        return self.convert(self.interface.get(False),isFat)

    def exclusiveJets(self,r =0.1,isFat = True):
        self.interface.makeExclusiveJets(r)
        return self.convert(self.interface.get(False),isFat)

    def exclusiveJetsUpTo(self,N=2,isFat = True ):
        self.interface.makeExclusiveJetsUpTo(N)
        return self.convert(self.interface.get(False),isFat)






#from DataFormats.FWLite import Events, Handle

#pfH = Handle('std::vector<pat::PackedCandidate')
#jetsH = Handle('std::vector<pat::Jet')

#events=Events(
#'root://eoscms//eos/cms/store/cmst3/user/bachtis/CMG/vv.root'
#)



#for ev in events:
#    ev.getByLabel('packedPFCandidates',pfH)
#    ev.getByLabel('slimmedJetsAK8',jetsH)
#    pf = pfH.product()
#    pfCHS = filter(lambda x: x.fromPV(0) , pf)
#    jetsDefault = jetsH.product()
#    if len(jetsDefault)==0:
#        continue
#    toolbox  = PyJetToolbox(pfCHS)
#    toolbox.setInterface(True,-1.0,0.8)
#    toolbox.setMassDrop(True)
#    toolbox.setSubjets(True,'inc',2)
#    toolbox.setPruning(False)
#    toolbox.setNtau(True)
#    toolbox.setSoftDrop(True)
#    jets=toolbox.inclusiveJets(100.0)
#    import pdb;pdb.set_trace()
       




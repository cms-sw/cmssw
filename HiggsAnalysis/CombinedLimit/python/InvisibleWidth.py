from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os


class InvisibleWidth(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.floatMass = True
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrama for Higgs mass range defined with inverterd order. Second must be larger the first"
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        self.modelBuilder.doVar("kV[1,0.0,1.1]") # bounded to 1
        self.modelBuilder.doVar("ktau[1,0.0,4.0]")
        self.modelBuilder.doVar("ktop[1,0.0,5.0]")
        self.modelBuilder.doVar("kbottom[1,0.0,4.0]")
        self.modelBuilder.doVar("kgluon[1,0.0,3.0]")
        self.modelBuilder.doVar("kgamma[1,0.0,3.0]")
        self.modelBuilder.doVar("BRInvUndet[0,0,1]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'kV,ktau,ktop,kbottom,kgluon,kgamma,BRInvUndet,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'kV,ktau,ktop,kbottom,kgluon,kgamma,BRInvUndet')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):

        # SM BR
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hzg", "hmm", "hss" ]: self.SMH.makeBR(d)

        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::invisibleWidth_Gscal_Vectors("@0*@0 * (@1+@2)", kV, SM_BR_hzz, SM_BR_hww)') 
        self.modelBuilder.factory_('expr::invisibleWidth_Gscal_tau("@0*@0 * (@1+@2)", ktau, SM_BR_htt, SM_BR_hmm)') 
        self.modelBuilder.factory_('expr::invisibleWidth_Gscal_top("@0*@0 * (@1+@2)", ktop, SM_BR_htoptop, SM_BR_hcc)')
        self.modelBuilder.factory_('expr::invisibleWidth_Gscal_bottom("@0*@0 * (@1+@2)", kbottom, SM_BR_hbb, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::invisibleWidth_Gscal_gluon("@0*@0 * @1", kgluon, SM_BR_hgluglu)')
        self.modelBuilder.factory_('expr::invisibleWidth_Gscal_gamma("@0*@0 * (@1+@2)", kgamma, SM_BR_hgg, SM_BR_hzg)')
        #self.modelBuilder.factory_('sum::invisibleWidth_Gscal_tot(invisibleWidth_Gscal_Vectors, invisibleWidth_Gscal_tau, invisibleWidth_Gscal_top, invisibleWidth_Gscal_bottom, invisibleWidth_Gscal_gluon, invisibleWidth_Gscal_gamma)')
        self.modelBuilder.factory_('expr::invisibleWidth_Gscal_tot("(@1+@2+@3+@4+@5+@6)/(1-@0)",BRInvUndet,invisibleWidth_Gscal_Vectors, invisibleWidth_Gscal_tau, invisibleWidth_Gscal_top, invisibleWidth_Gscal_bottom, invisibleWidth_Gscal_gluon, invisibleWidth_Gscal_gamma)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_('expr::invisibleWidth_BRscal_hvv("@0*@0/@1", kV, invisibleWidth_Gscal_tot)')
        self.modelBuilder.factory_('expr::invisibleWidth_BRscal_htt("@0*@0/@1", ktau, invisibleWidth_Gscal_tot)')
        self.modelBuilder.factory_('expr::invisibleWidth_BRscal_hbb("@0*@0/@1", kbottom, invisibleWidth_Gscal_tot)')
        self.modelBuilder.factory_('expr::invisibleWidth_BRscal_hgg("@0*@0/@1", kgamma, invisibleWidth_Gscal_tot)')

    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "invisibleWidth_XSBRscal_%s_%s" % (production,decay)
        print name, production, decay, energy
        if self.modelBuilder.out.function(name) == None:
            XSscal = "kgluon"
            if production in ["WH","ZH","VH","qqH"]: XSscal = "kV" 
            if production == "ttH": XSscal = "ktop"
            if decay == "hinv":
                self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, BRInvUndet)' % (name, XSscal))
            else:
                if decay in ["hbb", "htt", "hgg"]: BRscal = decay
                elif decay in ["hww", "hzz"]: BRscal = "hvv"
                else: print "Unknown decay mode:", decay
                self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, invisibleWidth_BRscal_%s)' % (name, XSscal, BRscal))
        return name


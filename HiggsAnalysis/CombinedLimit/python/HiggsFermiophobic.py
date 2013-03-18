from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os

class FermiophobicHiggs(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.mHRange = ['115','135'] # default
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrama for Higgs mass range defined with inverterd order. Second must be larger the first"
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("r[1,0,20]")
        if self.modelBuilder.out.var("MH"):
            self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
            self.modelBuilder.out.var("MH").setConstant(False)
        else:
            self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
        self.modelBuilder.doSet("POI",'r,MH')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
                
    def setup(self):
        ## Add FP BRs
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        self.SMH.textToSpline( 'FP_BR_hww', os.path.join(datadir, 'fp/BR.txt'), ycol=4 );
        self.SMH.textToSpline( 'FP_BR_hzz', os.path.join(datadir, 'fp/BR.txt'), ycol=5 );
        self.SMH.textToSpline( 'FP_BR_hgg', os.path.join(datadir, 'fp/BR.txt'), ycol=2 );
        self.SMH.textToSpline( 'FP_BR_hzg', os.path.join(datadir, 'fp/BR.txt'), ycol=3 );
        
        for decay in ['hww','hzz','hgg','hzg']:
            self.SMH.makeBR(decay)
            self.modelBuilder.factory_('expr::FP_BRScal_%s("@0*@1/@2",r,FP_BR_%s,SM_BR_%s)'%(decay,decay,decay))
        
        self.modelBuilder.out.Print()
    def getHiggsSignalYieldScale(self,production,decay,energy):
        if production not in ['VH', 'WH', 'ZH', 'qqH']:
            return 0
        if decay not in ['hww','hzz','hgg','hzg']:
            return 0       
        return 'FP_BRScal_%s' % decay




fp = FermiophobicHiggs()

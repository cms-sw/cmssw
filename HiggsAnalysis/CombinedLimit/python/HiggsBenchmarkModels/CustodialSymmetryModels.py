from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os


class LambdaWZHiggs(SMLikeHiggsModel):
    def __init__(self):
        SMLikeHiggsModel.__init__(self) 
        self.floatMass = False        
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.floatMass = True
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        self.modelBuilder.doVar("kZ[1,0,2]")
        self.modelBuilder.doVar("kf[1,0,2]")
        self.modelBuilder.doVar("lambdaWZ[1,0,2]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'kZ,lambdaWZ,kf,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'kZ,lambdaWZ,kf')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):

        self.decayScaling = {
            'hgg':'hgg',
            'hZg':'hZg',
            'hww':'hww',
            'hzz':'hzz',
            'hbb':'hff',
            'htt':'hff',
            }
        self.productionScaling = {
            'ggH':'kf',
            'ttH':'kf',
            'WH':'kW',
            'ZH':'kZ',
            }

        
        # define kW as lambdaWZ*kZ
        self.modelBuilder.factory_('expr::kW("@0*@1",kZ, lambdaWZ)')

        # scalings of the loops
        self.SMH.makeScaling('hgg', Cb='kf', Ctop='kf', CW='kW', Ctau='kf')
        self.SMH.makeScaling('hZg', Cb='kf', Ctop='kf', CW='kW', Ctau='kf')
        self.SMH.makeScaling('qqH', CW='kW', CZ='kZ')
        
        # SM BR
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)

        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::lambdaWZ_Gscal_Z("@0*@0 * @1", kZ, SM_BR_hzz)') 
        self.modelBuilder.factory_('expr::lambdaWZ_Gscal_W("@0*@0 * @1", kW, SM_BR_hww)') 
        self.modelBuilder.factory_('expr::lambdaWZ_Gscal_fermions("@0*@0 * (@1+@2+@3+@4+@5+@6+@7)", kf, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::lambdaWZ_Gscal_gg("@0 * @1", Scaling_hgg, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::lambdaWZ_Gscal_Zg("@0 * @1", Scaling_hZg, SM_BR_hZg)') 
        self.modelBuilder.factory_('sum::lambdaWZ_Gscal_tot(lambdaWZ_Gscal_Z, lambdaWZ_Gscal_W, lambdaWZ_Gscal_fermions, lambdaWZ_Gscal_gg, lambdaWZ_Gscal_Zg)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM) 
        self.modelBuilder.factory_('expr::lambdaWZ_BRscal_hzz("@0*@0/@1", kZ, lambdaWZ_Gscal_tot)')
        self.modelBuilder.factory_('expr::lambdaWZ_BRscal_hww("@0*@0/@1", kW, lambdaWZ_Gscal_tot)')
        self.modelBuilder.factory_('expr::lambdaWZ_BRscal_hff("@0*@0/@1", kf, lambdaWZ_Gscal_tot)')
        self.modelBuilder.factory_('expr::lambdaWZ_BRscal_hgg("@0/@1", Scaling_hgg, lambdaWZ_Gscal_tot)')
        self.modelBuilder.factory_('expr::lambdaWZ_BRscal_hZg("@0/@1", Scaling_hZg, lambdaWZ_Gscal_tot)')

        # verbosity
        #self.modelBuilder.out.Print()

    def getHiggsSignalYieldScale(self,production,decay,energy):

        name = 'lambdaWZ_XSBRscal_%(production)s_%(decay)s' % locals()
        
        #Special case that depends on Energy
        if production == 'qqH':
            self.productionScaling[production]='Scaling_qqH_'+energy
            name += '_%(energy)s' % locals()

        if self.modelBuilder.out.function(name):
            return name

        try:
            BRscal = self.decayScaling[decay]
            XSscal = self.productionScaling[production]
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, lambdaWZ_BRscal_%s)' % (name, XSscal, BRscal))
            return name
        except KeyError:
            if production == "VH":
                print "WARNING: You are trying to use a VH production mode in a model that needs WH and ZH separately. "\
                "The best I can do is to scale [%(production)s, %(decay)s, %(energy)s] with the decay BR only but this is wrong..." % locals()
                self.modelBuilder.factory_('expr::%s("1.0*@0", lambdaWZ_BRscal_%s)' % (name, BRscal))
                return name
            raise
        
class RzwHiggs(SMLikeHiggsModel):
    "scale WW by mu and ZZ by cZW^2 * mu"
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
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("Rzw[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Rzw,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Rzw')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
        
    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.doVar("Rhww[1,0,10]")
        self.modelBuilder.factory_('expr::Rhzz("@0*@1", Rhww, Rzw)')               
        
    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        else:
            return 'R%s' % decay

class RwzHiggs(SMLikeHiggsModel):
    "scale WW by mu and ZZ by cZW^2 * mu"
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
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("Rwz[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Rwz,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Rwz')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
        
    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.doVar("Rhzz[1,0,10]")
        self.modelBuilder.factory_('expr::Rhww("@0*@1", Rhzz, Rwz)')
               
        
    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        else:
            return 'R%s' % decay



class CzwHiggs(SMLikeHiggsModel):
    "Scale w and z and touch nothing else"
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
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("Czw[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Czw,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Czw')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.doVar("Cw[1,0,10]")
        self.modelBuilder.factory_('expr::Cz("@0*@1",Cw, Czw)')
            
        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::Czw_Gscal_tot("@0*@1 + @2*@3 + (1.0-@1-@3)", \
                                   Cw, SM_BR_hww, Cz, SM_BR_hzz)')
        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM) 
        self.modelBuilder.factory_('expr::Czw_BRscal_hww("@0/@1", Cw, Czw_Gscal_tot)')
        self.modelBuilder.factory_('expr::Czw_BRscal_hzz("@0/@1", Cz, Czw_Gscal_tot)')
        
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        for e in ['7TeV', '8TeV']:
            print 'build for %s'%e
            self.SMH.textToSpline(   'RqqH_%s'%e, os.path.join(datadir, 'couplings/R_VBF_%s.txt'%e), ycol=1 );
            self.modelBuilder.factory_('expr::Czw_XSscal_qqH_%s("(@0 + @1*@2) / (1.0 + @2) ", Cw, Cz, RqqH_%s)'%(e,e))
            self.modelBuilder.factory_('expr::Czw_XSscal_WH_%s("@0", Cw)'%e)
            self.modelBuilder.factory_('expr::Czw_XSscal_ZH_%s("@0", Cz)'%e)
            self.SMH.makeXS('WH',e)
            self.SMH.makeXS('ZH',e)
            self.modelBuilder.factory_('expr::Czw_XSscal_VH_%s("(@0*@1 + @2*@3) / (@1 + @3) ", Cw, SM_XS_WH_%s, Cz, SM_XS_ZH_%s)'%(e,e,e))

    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        
        name = "Czw_XSBRscal_%s_%s_%s" % (production,decay,energy)
        if self.modelBuilder.out.function(name) == None: 
            if production in ["ggH","ttH"]:
                self.modelBuilder.factory_('expr::%s("@0", Czw_BRscal_%s)' % (name, decay))
            else:
                self.modelBuilder.factory_('expr::%s("@0 * @1", Czw_XSscal_%s_%s, Czw_BRscal_%s)' % (name, production, energy, decay))
        return name


class CwzHiggs(SMLikeHiggsModel):
    "Scale w and z and touch nothing else"
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
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        self.modelBuilder.doVar("Cz[1,0,10]")
        self.modelBuilder.doVar("Cwz[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Cwz,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Cwz')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_('expr::Cw("@0*@1",Cz, Cwz)')
            
        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::Cwz_Gscal_tot("@0*@1 + @2*@3 + (1.0-@1-@3)", \
                                   Cw, SM_BR_hww, Cz, SM_BR_hzz)')
        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM) 
        self.modelBuilder.factory_('expr::Cwz_BRscal_hww("@0/@1", Cw, Cwz_Gscal_tot)')
        self.modelBuilder.factory_('expr::Cwz_BRscal_hzz("@0/@1", Cz, Cwz_Gscal_tot)')
        
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        for e in ['7TeV', '8TeV']:
            print 'build for %s'%e
            self.SMH.textToSpline(   'RqqH_%s'%e, os.path.join(datadir, 'couplings/R_VBF_%s.txt'%e), ycol=1 );
            self.modelBuilder.factory_('expr::Cwz_XSscal_qqH_%s("(@0 + @1*@2) / (1.0 + @2) ", Cw, Cz, RqqH_%s)'%(e,e))
            self.modelBuilder.factory_('expr::Cwz_XSscal_WH_%s("@0", Cw)'%e)
            self.modelBuilder.factory_('expr::Cwz_XSscal_ZH_%s("@0", Cz)'%e)
            self.SMH.makeXS('WH',e)
            self.SMH.makeXS('ZH',e)
            self.modelBuilder.factory_('expr::Cwz_XSscal_VH_%s("(@0*@1 + @2*@3) / (@1 + @3) ", Cw, SM_XS_WH_%s, Cz, SM_XS_ZH_%s)'%(e,e,e))

    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        
        name = "Cwz_XSBRscal_%s_%s_%s" % (production,decay,energy)
        if self.modelBuilder.out.function(name) == None: 
            if production in ["ggH","ttH"]:
                self.modelBuilder.factory_('expr::%s("@0", Cwz_BRscal_%s)' % (name, decay))
            else:
                self.modelBuilder.factory_('expr::%s("@0 * @1", Cwz_XSscal_%s_%s, Cwz_BRscal_%s)' % (name, production, energy, decay))
        return name

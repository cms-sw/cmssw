from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os

class MepsHiggs(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
        self.MRange = ['150','350']
        self.epsRange = ['-1','1']
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
            if po.startswith("MRange="):
                self.MRange = po.replace("MRange=","").split(":")
                if len(self.MRange) != 2:
                    raise RuntimeError, "M range requires minimal and maximal value"
                elif float(self.MRange[0]) >= float(self.MRange[1]):
                    raise RuntimeError, "minimal and maximal range swapped. Second value must be larger first one"
            if po.startswith("epsRange="):
                self.epsRange = po.replace("epsRange=","").split(":")
                if len(self.epsRange) != 2:
                    raise RuntimeError, "epsilon range requires minimal and maximal value"
                elif float(self.epsRange[0]) >= float(self.epsRange[1]):
                    raise RuntimeError, "minimal and maximal range swapped. Second value must be larger first one"
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("M[246.22,%s,%s]" % (self.MRange[0], self.MRange[1]))
        self.modelBuilder.doVar("eps[0,%s,%s]" % (self.epsRange[0], self.epsRange[1]))

        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'M,eps,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'M,eps')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
        
    def setup(self):

        self.modelBuilder.doVar("SM_VEV[246.22]")
        self.msbar = {
            'top' : (160, (-4.3,+4.8)),
            'b'   : (4.18, (-0.03,+0.03)),
            'tau' : (1.77682, (-0.16,+0.16)),
            'mu'  : (0.105658, (-0.0000035,+0.0000035)),
            'W'   : (80.385, (-0.015,+0.015)),
            'Z'   : (91.1876, (-0.0021,+0.0021)),
            }

        for name, vals in self.msbar.iteritems():
            #self.modelBuilder.doVar("M%s_MSbar[%s,%s,%s]" % (name, vals[0], vals[0]+vals[1][0], vals[0]+vals[1][1]))
            self.modelBuilder.doVar("M%s_MSbar[%s]" % (name, vals[0]))

            if name in ('W','Z'):
                # cv == v (mv^(2 e)/M^(1 + 2 e))
                self.modelBuilder.factory_(
                    'expr::C%(name)s("@0 * TMath::Power(@3,2*@2) / TMath::Power(@1,1+2*@2)", SM_VEV, M, eps, M%(name)s_MSbar)' % locals() )
            else:
                # cf == v (mf^e/M^(1 + e))
                self.modelBuilder.factory_(
                    'expr::C%(name)s("@0 * TMath::Power(@3,@2) / TMath::Power(@1,1+@2)", SM_VEV, M, eps, M%(name)s_MSbar)' % locals() )
        
        self.productionScaling = {
            'ttH':'Ctop',
            'WH':'CW',
            'ZH':'CZ',
            }

        self.SMH.makeScaling('ggH', Cb='Cb', Ctop='Ctop')
        self.SMH.makeScaling('qqH', CW='CW', CZ='CZ')

        self.SMH.makeScaling('hgluglu', Cb='Cb', Ctop='Ctop')
        
        self.SMH.makeScaling('hgg', Cb='Cb', Ctop='Ctop', CW='CW', Ctau='Ctau')
        self.SMH.makeScaling('hzg', Cb='Cb', Ctop='Ctop', CW='CW', Ctau='Ctau')
        
        ## partial widths, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hzg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_('expr::Meps_Gscal_w("@0*@0 * @1", CW, SM_BR_hww)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_z("@0*@0 * @1", CZ, SM_BR_hzz)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_b("@0*@0 * @1", Cb, SM_BR_hbb)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_tau("@0*@0 * @1", Ctau, SM_BR_htt)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_mu("@0*@0 * @1", Cmu, SM_BR_hmm)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_top("@0*@0 * @1", Ctop, SM_BR_htoptop)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_glu("@0 * @1", Scaling_hgluglu, SM_BR_hgluglu)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_gg("@0 * @1", Scaling_hgg, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::Meps_Gscal_zg("@0 * @1", Scaling_hzg, SM_BR_hzg)') 
        self.modelBuilder.factory_('sum::Meps_Gscal_tot(Meps_Gscal_w, Meps_Gscal_z, Meps_Gscal_b, Meps_Gscal_tau, Meps_Gscal_mu, Meps_Gscal_top, Meps_Gscal_glu, Meps_Gscal_gg, Meps_Gscal_zg, SM_BR_hcc, SM_BR_hss)')
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_('expr::Meps_BRscal_hww("@0*@0/@1", CW, Meps_Gscal_tot)')
        self.modelBuilder.factory_('expr::Meps_BRscal_hzz("@0*@0/@1", CZ, Meps_Gscal_tot)')
        self.modelBuilder.factory_('expr::Meps_BRscal_hbb("@0*@0/@1", Cb, Meps_Gscal_tot)')
        self.modelBuilder.factory_('expr::Meps_BRscal_htt("@0*@0/@1", Ctau, Meps_Gscal_tot)')
        self.modelBuilder.factory_('expr::Meps_BRscal_hmm("@0*@0/@1", Cmu, Meps_Gscal_tot)')
        self.modelBuilder.factory_('expr::Meps_BRscal_hgg("@0/@1", Scaling_hgg, Meps_Gscal_tot)')
        self.modelBuilder.factory_('expr::Meps_BRscal_hzg("@0/@1", Scaling_hzg, Meps_Gscal_tot)')
        
        self.modelBuilder.out.Print()

    def getHiggsSignalYieldScale(self,production,decay,energy):
    
        name = 'Meps_XSBRscal_%(production)s_%(decay)s' % locals()
        
        if production in ('ggH','qqH'):
            self.productionScaling[production]='Scaling_'+production+'_'+energy
            name += '_%(energy)s' % locals()
            
        if self.modelBuilder.out.function(name):
            return name

        if production == "VH":
            print "WARNING: You are trying to use a VH production mode in a model that needs WH and ZH separately. "\
            "The best I can do is to scale [%(production)s, %(decay)s, %(energy)s] with the decay BR only but this is wrong..." % locals()
            self.modelBuilder.factory_('expr::%(name)s("1.0*@0", Meps_BRscal_%(decay)s)' % locals())
            return name

        XSscal = self.productionScaling[production]
        if production == 'ggH':
            self.modelBuilder.factory_('expr::%(name)s("@0 * @1", %(XSscal)s, Meps_BRscal_%(decay)s)' % locals())
        else:
            self.modelBuilder.factory_('expr::%(name)s("@0*@0 * @1", %(XSscal)s, Meps_BRscal_%(decay)s)' % locals())
        return name

class ResolvedC6(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
        self.MRange = ['150','350']
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
            if po.startswith("MRange="):
                self.MRange = po.replace("MRange=","").split(":")
                if len(self.MRange) != 2:
                    raise RuntimeError, "M range requires minimal and maximal value"
                elif float(self.MRange[0]) >= float(self.MRange[1]):
                    raise RuntimeError, "minimal and maximal range swapped. Second value must be larger first one"
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("CW[1.0,0.0,5.0]")
        self.modelBuilder.doVar("CZ[1.0,0.0,5.0]")
        self.modelBuilder.doVar("Ctop[1.0,0.0,5.0]")
        self.modelBuilder.doVar("Cb[1.0,0.0,5.0]")
        self.modelBuilder.doVar("Ctau[1.0,0.0,5.0]")
        self.modelBuilder.doVar("Cmu[1.0,0.0,5.0]")

        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'MH,CW,CZ,Ctop,Cb,Ctau,Cmu')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'CW,CZ,Ctop,Cb,Ctau,Cmu')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
        
    def setup(self):

        self.productionScaling = {
            'ttH':'Ctop',
            'WH':'CW',
            'ZH':'CZ',
            }

        self.SMH.makeScaling('ggH', Cb='Cb', Ctop='Ctop')
        self.SMH.makeScaling('qqH', CW='CW', CZ='CZ')

        self.SMH.makeScaling('hgluglu', Cb='Cb', Ctop='Ctop')
        
        self.SMH.makeScaling('hgg', Cb='Cb', Ctop='Ctop', CW='CW', Ctau='Ctau')
        self.SMH.makeScaling('hzg', Cb='Cb', Ctop='Ctop', CW='CW', Ctau='Ctau')
        
        ## partial widths, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hzg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_('expr::wztbtm_Gscal_w("@0*@0 * @1", CW, SM_BR_hww)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_z("@0*@0 * @1", CZ, SM_BR_hzz)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_b("@0*@0 * @1", Cb, SM_BR_hbb)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_tau("@0*@0 * @1", Ctau, SM_BR_htt)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_mu("@0*@0 * @1", Cmu, SM_BR_hmm)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_top("@0*@0 * @1", Ctop, SM_BR_htoptop)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_glu("@0 * @1", Scaling_hgluglu, SM_BR_hgluglu)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_gg("@0 * @1", Scaling_hgg, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::wztbtm_Gscal_zg("@0 * @1", Scaling_hzg, SM_BR_hzg)') 
        self.modelBuilder.factory_('sum::wztbtm_Gscal_tot(wztbtm_Gscal_w, wztbtm_Gscal_z, wztbtm_Gscal_b, wztbtm_Gscal_tau, wztbtm_Gscal_mu, wztbtm_Gscal_top, wztbtm_Gscal_glu, wztbtm_Gscal_gg, wztbtm_Gscal_zg, SM_BR_hcc, SM_BR_hss)')
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_('expr::wztbtm_BRscal_hww("@0*@0/@1", CW, wztbtm_Gscal_tot)')
        self.modelBuilder.factory_('expr::wztbtm_BRscal_hzz("@0*@0/@1", CZ, wztbtm_Gscal_tot)')
        self.modelBuilder.factory_('expr::wztbtm_BRscal_hbb("@0*@0/@1", Cb, wztbtm_Gscal_tot)')
        self.modelBuilder.factory_('expr::wztbtm_BRscal_htt("@0*@0/@1", Ctau, wztbtm_Gscal_tot)')
        self.modelBuilder.factory_('expr::wztbtm_BRscal_hmm("@0*@0/@1", Cmu, wztbtm_Gscal_tot)')
        self.modelBuilder.factory_('expr::wztbtm_BRscal_hgg("@0/@1", Scaling_hgg, wztbtm_Gscal_tot)')
        self.modelBuilder.factory_('expr::wztbtm_BRscal_hzg("@0/@1", Scaling_hzg, wztbtm_Gscal_tot)')
        
        self.modelBuilder.out.Print()

    def getHiggsSignalYieldScale(self,production,decay,energy):
    
        name = 'wztbtm_XSBRscal_%(production)s_%(decay)s' % locals()
        
        if production in ('ggH','qqH'):
            self.productionScaling[production]='Scaling_'+production+'_'+energy
            name += '_%(energy)s' % locals()
            
        if self.modelBuilder.out.function(name):
            return name

        if production == "VH":
            print "WARNING: You are trying to use a VH production mode in a model that needs WH and ZH separately. "\
            "The best I can do is to scale [%(production)s, %(decay)s, %(energy)s] with the decay BR only but this is wrong..." % locals()
            self.modelBuilder.factory_('expr::%(name)s("1.0*@0", wztbtm_BRscal_%(decay)s)' % locals())
            return name

        XSscal = self.productionScaling[production]
        if production == 'ggH':
            self.modelBuilder.factory_('expr::%(name)s("@0 * @1", %(XSscal)s, wztbtm_BRscal_%(decay)s)' % locals())
        else:
            self.modelBuilder.factory_('expr::%(name)s("@0*@0 * @1", %(XSscal)s, wztbtm_BRscal_%(decay)s)' % locals())
        return name


mEpsHiggs = MepsHiggs()
resolvedC6 = ResolvedC6()

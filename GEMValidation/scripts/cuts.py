from ROOT import *

#_______________________________________________________________________________
def ANDtwo(cut1,cut2):
    """AND of two TCuts in PyROOT"""
    return TCut("(%s) && (%s)"%(cut1.GetTitle(),cut2.GetTitle()))


#_______________________________________________________________________________
def ORtwo(cut1,cut2):
    """OR of two TCuts in PyROOT"""
    return TCut("(%s) || (%s)"%(cut1.GetTitle(),cut2.GetTitle()))


#_______________________________________________________________________________
def AND(*arg):
    """AND of any number of TCuts in PyROOT"""
    length = len(arg)
    if length == 0:
        print "ERROR: invalid number of arguments"
        return
    if length == 1:
        return arg[0] 
    if length==2:
        return ANDtwo(arg[0],arg[1])
    if length>2:
        result = arg[0]
        for i in range(1,len(arg)):
            result = ANDtwo(result,arg[i])
        return result


#_______________________________________________________________________________
def OR(*arg): 
    """OR of any number of TCuts in PyROOT"""
    length = len(arg)
    if length == 0:
        print "ERROR: invalid number of arguments"
        return
    if length == 1:
        return arg[0] 
    if length==2:
        return ORtwo(arg[0],arg[1])
    if length>2:
        result = arg[0]
        for i in range(1,len(arg)):
            result = ORtwo(result,arg[i])
        return result


#_______________________________________________________________________________
nocut = TCut("")

muon = TCut("TMath::Abs(particleType)==13")
nonMuon = TCut("TMath::Abs(particleType)!=13")
all = OR(muon,nonMuon)

rm1 = TCut("region==-1")
rp1 = TCut("region==1")

even = TCut("chamber%2==0")
odd  = TCut("chamber%2==1")

l1 = TCut("layer==1")
l2 = TCut("layer==2")
l3 = TCut("layer==3")
l4 = TCut("layer==4")
l5 = TCut("layer==5")
l6 = TCut("layer==6")

st1 = TCut("station==1")
st2 = TCut("station==2")
st3 = TCut("station==3")

eta_min = 1.64
eta_max = 2.12

ok_eta_min = TCut("TMath::Abs(eta) > %f"%(eta_min))
ok_eta_max = TCut("TMath::Abs(eta) < %f"%(eta_max))
ok_eta = AND(ok_eta_min,ok_eta_max)
ok_gL1sh = TCut("gem_sh_layer1 > 0")
ok_gL2sh = TCut("gem_sh_layer2 > 0")
ok_gL1dg = TCut("gem_dg_layer1 > 0")
ok_gL2dg = TCut("gem_dg_layer2 > 0")
ok_gL1pad = TCut("gem_pad_layer1 > 0")
ok_gL2pad = TCut("gem_pad_layer2 > 0")
ok_gL1rh = TCut("gem_rh_layer1 > 0")
ok_gL2rh = TCut("gem_rh_layer2 > 0")

ok_trk_gL1sh = TCut("has_gem_sh_l1 > 0")
ok_trk_gL2sh = TCut("has_gem_sh_l2 > 0")

ok_trk_gL1dg = TCut("has_gem_dg_l1 > 0")
ok_trk_gL2dg = TCut("has_gem_dg_l2 > 0")

ok_lx_odd =  TCut("TMath::Abs(TMath::ASin(gem_lx_odd/gem_trk_rho)) < 5*TMath::Pi()/180.")
ok_lx_even = TCut("TMath::Abs(TMath::ASin(gem_lx_even/gem_trk_rho)) < 5*TMath::Pi()/180.")

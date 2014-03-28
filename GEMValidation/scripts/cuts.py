from ROOT import *

#_______________________________________________________________________________
def ANDtwo(cut1,cut2):
    """AND of two TCuts in PyROOT"""
    if cut1.GetTitle() == "":
        return cut2
    if cut2.GetTitle() == "":
        return cut1
    return TCut("(%s) && (%s)"%(cut1.GetTitle(),cut2.GetTitle()))


#_______________________________________________________________________________
def ORtwo(cut1,cut2):
    """OR of two TCuts in PyROOT"""
    if cut1.GetTitle() == "":
        return cut2
    if cut2.GetTitle() == "":
        return cut1
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
def la(i):
    return TCut("layer==%d"%(i))


#_______________________________________________________________________________
def st(i):
    return TCut("station==%d"%(i))


#_______________________________________________________________________________
def ri(i):
    return TCut("ring==%d"%(i))


#_______________________________________________________________________________
def ch(i):
    return TCut("chamber==%d"%(i))


#_______________________________________________________________________________
nocut = TCut("")

muon = TCut("TMath::Abs(particleType)==13")
nonMuon = TCut("TMath::Abs(particleType)!=13")
all = OR(muon,nonMuon)

rm1 = TCut("region==-1")
rp1 = TCut("region==1")

ec2 = TCut("endcap==2") 
ec1 = TCut("endcap==1")

even = TCut("chamber%2==0")
odd  = TCut("chamber%2==1")

rpc_sector_even = TCut("sector%2==0")
rpc_sector_odd  = TCut("sector%2==1")

rpc_subsector_even = TCut("subsector%2==0")
rpc_subsector_odd  = TCut("subsector%2==1")

l1 = TCut("layer==1")
l2 = TCut("layer==2")
l3 = TCut("layer==3")
l4 = TCut("layer==4")
l5 = TCut("layer==5")
l6 = TCut("layer==6")

st1 = TCut("station==1")
st2 = TCut("station==2")
st3 = TCut("station==3")
st4 = TCut("station==4")

ri1 = TCut("ring==1")
ri2 = TCut("ring==2")
ri3 = TCut("ring==3")

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

## CSC simhits & digis
ok_sh1 = TCut("(has_csc_sh&1) > 0")
ok_sh2 = TCut("(has_csc_sh&2) > 0")
ok_st1 = TCut("(has_csc_strips&1) > 0")
ok_st2 = TCut("(has_csc_strips&2) > 0")
ok_w1 = TCut("(has_csc_wires&1) > 0")
ok_w2 = TCut("(has_csc_wires&2) > 0")
ok_digi1 = AND(ok_st1,ok_w1)
ok_digi2 = AND(ok_st2,ok_w2)

## CSC stub
ok_lct1 = TCut("(has_lct&1) > 0")
ok_lct2 = TCut("(has_lct&2) > 0")
ok_alct1 = TCut("(has_alct&1) > 0")
ok_alct2 = TCut("(has_alct&2) > 0")
ok_clct1 = TCut("(has_clct&1) > 0")
ok_clct2 = TCut("(has_clct&2) > 0")
ok_lct_hs_min = TCut("hs_lct_odd > 4")
ok_lct_hs_max = TCut("hs_lct_odd < 125")
ok_lct_hs = AND(ok_lct_hs_min,ok_lct_hs_max)
ok_lcths1 = AND(ok_lct1,ok_lct_hs)
ok_lcths2 = AND(ok_lct2,ok_lct_hs)

## GEM simhit
ok_gsh1 = TCut("(has_gem_sh&1) > 0")
ok_gsh2 = TCut("(has_gem_sh&2) > 0")
ok_g2sh1 = TCut("(has_gem_sh2&1) > 0")
ok_g2sh2 = TCut("(has_gem_sh2&2) > 0")


## GEM digi
ok_gdg1 = TCut("(has_gem_dg&1) > 0")
ok_gdg2 = TCut("(has_gem_dg&2) > 0")
ok_pad1 = TCut("(has_gem_pad&1) > 0")
ok_pad2 = TCut("(has_gem_pad&2) > 0")

ok_dphi1 = TCut("dphi_pad_odd < 10.")
ok_dphi2 = TCut("dphi_pad_even < 10.")

ok_pad1_lct1 = AND(ok_pad1,ok_lct1)
ok_pad2_lct2 = AND(ok_pad2,ok_lct2)

ok_pad1_dphi1 = AND(ok_pad1,ok_dphi1)
ok_pad2_dphi2 = AND(ok_pad2,ok_dphi2)

ok_lct1_eta = AND(ok_eta,ok_lct1)
ok_lct2_eta = AND(ok_eta,ok_lct2)

ok_pad1_lct1_eta = AND(ok_pad1,ok_lct1,ok_eta)
ok_pad2_lct2_eta = AND(ok_pad2,ok_lct2,ok_eta)

ok_gsh1_lct1_eta = AND(ok_gsh1,ok_lct1,ok_eta)
ok_gsh2_lct2_eta = AND(ok_gsh2,ok_lct2,ok_eta)

ok_gsh1_eta = AND(ok_gsh1,ok_eta)
ok_gsh2_eta = AND(ok_gsh2,ok_eta)

ok_gdg1_eta = AND(ok_gdg1,ok_eta)
ok_gdg2_eta = AND(ok_gdg2,ok_eta)

ok_2pad1 = TCut("(has_gem_pad2&1) > 0")
ok_2pad2 = TCut("(has_gem_pad2&2) > 0")

ok_pad1_overlap = OR(ok_pad1,AND(ok_lct2,ok_pad2))
ok_pad2_overlap = OR(ok_pad2,AND(ok_lct1,ok_pad1))

ok_copad1 = TCut("(has_gem_copad&1) > 0")
ok_copad2 = TCut("(has_gem_copad&2) > 0")

ok_Qp = TCut("charge > 0")
ok_Qn = TCut("charge < 0")

ok_lct1_eta_Qn = AND(ok_lct1,ok_eta,ok_Qn)
ok_lct2_eta_Qn = AND(ok_lct2,ok_eta,ok_Qn)

ok_lct1_eta_Qp = AND(ok_lct1,ok_eta,ok_Qp)
ok_lct2_eta_Qp = AND(ok_lct2,ok_eta,ok_Qp)

Ep = TCut("endcap > 0")
En = TCut("endcap < 0")

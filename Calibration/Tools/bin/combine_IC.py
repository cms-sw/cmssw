from ROOT import *
gSystem.Load("libFWCoreFWLite.so")
AutoLibraryLoader.enable()
gSystem.Load("libCalibrationTools.so")

all = DSAll()
eb = DSIsBarrel()
eep = DSIsEndcapPlus()
eem = DSIsEndcapMinus()

rings = DRings()
rings.setEERings("eerings.dat")

ic1 = IC()
ic2 = IC()
ic_scale = IC()
res = IC()
res.setRings(rings)
icdiff = IC()

check = IC()

dir = "combination_120615"
file_scale = dir + "/ic_GR_R_52_V8.txt"
#file2 = dir + "/IC_phi_191043_191810.txt"
#filec = dir + "/IC_pizphi_191043_191810.txt"
#filec = dir + "/interCalib_GR_R_52_V8.txt"
file1 = dir + "/IC_piz_191043_191810_newerr_noeta60.txt"
file2 = dir + "/IC_phi_191043_191810_abserr.txt"
filec = dir + "/IC_pizphi_191043_191810_update.txt"

file1 = dir + "/IC_piz_193093_194897_newerr2.txt"
file2 = dir + "/IC_phi_193093_194897_abserr.txt"
filec = dir + "/IC_pizphi_193093_194897_update.txt"

#file1 = dir + "/IC_piz_191043_191810.txt"
#file2 = dir + "/IC_phi_191043_191810.txt"
#filec = dir + "/IC_pizphi_191043_191810.txt"

IC.readTextFile(file1, ic1)
IC.readTextFile(file2, ic2)
IC.readTextFile(filec, check)
IC.readTextFile(file_scale, ic_scale)

IC.scaleEta(check, check, True)
IC.scaleEta(check, ic_scale)
IC.dump(check, "phipiz.txt", all)

p = TProfile("p", "p", 800, -200, 200)

fout = TFile("combo_histos.root", "recreate")

p1 = p.Clone("p1")
p2 = p.Clone("p2")
pr = p.Clone("pr")
ps = p.Clone("ps")
pc = p.Clone("pc")
pd = p.Clone("pd")

IC.profileEta(ic1, p1, all)
IC.profileEta(ic2, p2, all)

IC.scaleEta(ic1, ic1, True)
IC.scaleEta(ic1, ic_scale)

IC.scaleEta(ic2, ic2, True)
IC.scaleEta(ic2, ic_scale)

IC.combine(ic1, ic2, res)
IC.scaleEta(res, res, True)
IC.scaleEta(res, ic_scale)

print "overall average:", IC.average(res, all)
print "     EB average:", IC.average(res, eb)
print "    EE+ average:", IC.average(res, eep)
print "    EE- average:", IC.average(res, eem)

IC.profileEta(res, pr, all)
IC.profileEta(ic_scale, ps, all)
IC.profileEta(check, pc, all)

IC.dump(res, "combined.txt", all)

IC.multiply(res, -1, res)
IC.add(res, check, icdiff)
IC.profileEta(icdiff, pd, all)
h = TH1F("h", "h", 1000, -.1, .1)
IC.constantDistribution(icdiff, h, all)

fout.Write()
#fout.Close()

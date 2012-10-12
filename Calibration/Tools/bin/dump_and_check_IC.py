import sys

if len(sys.argv) < 2:
        print "Usage: %s <IC file> [reference IC file]" % sys.argv[0]
        sys.exit(1)

from ROOT import *
import os
import string

def cat(filename, fileout):
        for l in open(filename):
                fileout.write(l)

def sep(file):
        file.write("---------------------------------\n")


def dump_h1(o, filename):
        f = open(filename, "w")
        f.write("# " + o.ClassName() + "\n")
        f.write("#nbin = %d; xmin = %.10g; xmax = %.10g;\n" % (o.GetNbinsX(), o.GetXaxis().GetXmin(), o.GetXaxis().GetXmax()))
        f.write("# x y dx dy\n")
        f.write("#\n")
        for i in range(1, o.GetNbinsX() + 1):
                f.write("%.10g %.10g %.10g %.10g\n" % (o.GetBinCenter(i), o.GetBinContent(i), o.GetBinWidth(i), o.GetBinError(i)))

 
gSystem.Load("libFWCoreFWLite.so")
AutoLibraryLoader.enable()
gSystem.Load("libCalibrationTools.so")

file = sys.argv[1]
bn = os.path.basename(file)
bn = string.replace(bn, ".xml", "")
bn = string.replace(bn, ".dat", "")

twofiles = 0

if len(sys.argv) > 2:
        file_ref = sys.argv[2]
        bn_ref = os.path.basename(file_ref)
        bn_ref = string.replace(bn_ref, ".xml", "")
        bn_ref = string.replace(bn_ref, ".dat", "")
        bn = bn + "__wref_" + bn_ref
        twofiles = 1

min, max = 0.6, 2.
emin, emax = 0.001, 0.10
#min, max = 0.8, 1.2

dir = "dumps_" + bn + "/"
os.system("mkdir -p " + dir)

rings = DRings()
rings.setEERings("eerings.dat")

ic = IC()
IC.readTextFile(file, ic)
#IC.readXMLFile(file, ic)
ic.setRings(rings)

if len(sys.argv) > 2:
        ic_ref = IC()
        if (string.find(file_ref, ".xml", len(file_ref) - 4, len(file_ref)) == -1):
                IC.readTextFile(file_ref, ic_ref)
        else:
                IC.readXMLFile(file_ref, ic_ref)
        IC.reciprocal(ic_ref, ic_ref)
        IC.multiply(ic, ic_ref, ic)

# DetId selectors
all = DSAll()
eb = DSIsBarrel()
ee = DSIsEndcap()
eep = DSIsEndcapPlus()
eem = DSIsEndcapMinus()

eun = DSIsUncalib()
eun.setIC(ic)

IC.dump(ic, dir + bn + ".dat", all)

eou = DSIsOutlier()
eou.setIC(ic)

# logfile
logname = dir + "log_" + bn + ".log"
log = open(logname, "w")

# histograms
pe = TProfile("pe", "pe", 600, -150, 150)
pp = TProfile("pp", "pp", 720, 0.5, 360.5)
psm = TProfile("psm", "psm", 36, 0.5, 36.5)
#pp = TProfile("pp", "pp", 560, -10, 370)
#meb = T2HF("h2eb", "h2eb", 360, 0.5, 360.5, 85, 0.5, 85.5)
#mee = T2HF("h2ee", "h2ee", 100, 0.5, 100.5, 100, 0.5, 100.5)
h = TH1F("h", "h", 250 * (1 + 9 * twofiles), 0, 5.)

frout = TFile.Open(dir + "histos_" + bn + ".root", "recreate")
hall = h.Clone("hall")
IC.constantDistribution(ic, hall, all)
heb = h.Clone("heb")
IC.constantDistribution(ic, heb, eb)
heem = h.Clone("heem")
IC.constantDistribution(ic, heem, eem)
heep = h.Clone("heep")
IC.constantDistribution(ic, heep, eep)

pe_all = pe.Clone("pe_all")
pe_all.BuildOptions(0, 0, "s")
IC.profileEta(ic, pe_all, all)
pe_all_err = pe.Clone("pe_all_err")
pe_all_err.BuildOptions(0, 0, "s")
IC.profileEta(ic, pe_all_err, all, True)

pp_all = pp.Clone("pe_all")
pp_all.BuildOptions(0, 0, "s")
IC.profilePhi(ic, pp_all, all)
pp_all_err = pp.Clone("pp_all_err")
pp_all_err.BuildOptions(0, 0, "s")
IC.profilePhi(ic, pp_all_err, all, True)

psm_all = psm.Clone("psm_all")
psm_all.BuildOptions(0, 0, "s")
IC.profileSM(ic, psm_all, all)
psm_all_err = psm.Clone("psm_all_err")
psm_all_err.BuildOptions(0, 0, "s")
IC.profileSM(ic, psm_all_err, all, True)

sep(log)

dump_h1(hall, dir + "dump_ic_distrib_all_" + bn + ".dat")
dump_h1(heb, dir + "dump_ic_distrib_eb_" + bn + ".dat")
dump_h1(heem, dir + "dump_ic_distrib_eem_" + bn + ".dat")
dump_h1(heep, dir + "dump_ic_distrib_eep_" + bn + ".dat")
dump_h1(pe_all, dir + "dump_profile_eta_" + bn + ".dat")
dump_h1(pe_all_err, dir + "dump_profile_eta_err_" + bn + ".dat")
dump_h1(pp_all, dir + "dump_profile_phi_" + bn + ".dat")
dump_h1(pp_all_err, dir + "dump_profile_phi_err_" + bn + ".dat")
dump_h1(psm_all, dir + "dump_profile_sm_" + bn + ".dat")
dump_h1(psm_all_err, dir + "dump_profile_sm_err_" + bn + ".dat")

IC.dumpEtaScale(ic, dir + "dump_etascale_" + bn + ".dat")

tmpfilename = dir + "dump_uncalibrated_" + bn + ".dat"
IC.dump(ic, tmpfilename, eun)
log.write("Non Calibrated\n")
cat(tmpfilename, log)
sep(log)

eou.setThresholds(min, max)
tmpfilename = dir + "dump_outliers_" + str(min) + "_" + str(max) + "_" + bn + ".dat"
IC.dump(ic, tmpfilename, eou)
log.write("Outliers:  ic < " + str(min) + "  ||  ic > " + str(max) + "\n")
cat(tmpfilename, log)
sep(log)

eou.setThresholds(emin, emax, True)
tmpfilename = dir + "dump_outliers_error_" + str(emin) + "_" + str(emax) + "_" + bn + ".dat"
IC.dump(ic, tmpfilename, eou)
log.write("Outliers by error:  eic < " + str(emin) + "  ||  eic > " + str(emax) + "\n")
cat(tmpfilename, log)
sep(log)

log.write("Averages\n")
log.write("  overall: %f\n" % IC.average(ic, all))
log.write("       EB: %f\n" % IC.average(ic, eb))
log.write("       EE: %f\n" % IC.average(ic, ee))
log.write("      EE+: %f\n" % IC.average(ic, eep))
log.write("      EE-: %f\n" % IC.average(ic, eem))
sep(log)

log.flush()

cat(logname, sys.stdout)

frout.Write()
frout.Close()

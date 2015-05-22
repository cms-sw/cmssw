from math import *
import os.path

for ana in [ "2lss_ee","2lss_mumu","2lss_em", "3l_tight", "4l" ]:
    jetbins = [ "" ] if "2lss" in ana else ['']
    for j in jetbins:
        btags = [ ('',''), ('_bt','.btight'), ('_btFRMC','.btight'), ('_bl','.bloose') ]
        for bpost, bname in btags:
            filename = "btagYields.%s%s%s.txt" % (ana,j,bname)
            if not os.path.exists(filename): continue
            file = open(filename,'r')
            lines = [ l.strip() for l in file ]
            if len(lines) != 3 or len(lines[0].split())-5 != len(lines[2].split())-3:
                print "Malformed file %s" % filename
                continue
            procs  = lines[0].split()[1:-4]
            yields = lines[2].split()[1:-2]
            ymap = dict([(p,float(y)) for p,y in zip(procs,yields)])
            for p in 'ttH','TTW','TTZ','WZ','ZZ':
                if p == 'ZZ' and "2lss" in ana: continue # negligible
                if p in ['TTW','WZ'] and "4l" in ana: continue # not from MC
                P=p.upper()
                kbup = ymap[P+"_btag_bUp"]/ymap[P] if ymap[P] != 0 else 1
                kbdn = ymap[P+"_btag_bDn"]/ymap[P] if ymap[P] != 0 else 1
                klup = ymap[P+"_btag_lUp"]/ymap[P] if ymap[P] != 0 else 1
                kldn = ymap[P+"_btag_lDn"]/ymap[P] if ymap[P] != 0 else 1
                binmap = "%s.*%s.*%s" % (ana,j,bpost) if j != '' else "%s.*%s" % (ana,bpost)
                binmap = binmap.replace("3l_tight","3l")
                if abs(kbup-1) < 0.015 and abs(kbdn-1) < 0.015:
                    print "CMS_ttHlep_eff_b  : %-20s : %-20s : - # negligible: %.2f/%.2f" % (p,binmap,kbdn,kbup)
                elif abs(kbup*kbdn-1) < 0.015:
                    print "CMS_ttHlep_eff_b  : %-20s : %-20s : %.2f" % (p,binmap,sqrt(kbup/kbdn))
                else:
                    print "CMS_ttHlep_eff_b  : %-20s : %-20s : %.2f/%.2f  # asymm: %+.3f" % (p,binmap,kbdn,kbup,kbup*kbdn-1)
                if abs(klup-1) < 0.015 and abs(kldn-1) < 0.015:
                    print "CMS_ttHlep_fake_b : %-20s : %-20s : - # negligible: %.2f/%.2f" % (p,binmap,kldn,klup)
                elif abs(klup*kldn-1) < 0.015:
                    print "CMS_ttHlep_fake_b : %-20s : %-20s : %.2f" % (p,binmap,sqrt(klup/kldn))
                else:
                    print "CMS_ttHlep_fake_b : %-20s : %-20s : %.2f/%.2f  # asymm: %+.3f" % (p,binmap,kldn,klup,klup*kldn-1)
    

//
// Federico Ferri, CEA-Saclay Irfu/SPP, 14.12.2011
// federico.ferri@cern.ch
//

#include "Calibration/Tools/interface/IC.h"

#include "TFile.h"
#include "TPad.h"
#include "TStyle.h"



int main()
{
        EcalChannelStatus chStatus;
        IC::readEcalChannelStatusFromTextFile("chStatus.dat", chStatus);

        DSIsEndcap isEndcap;
        DSIsBarrel isBarrel;

        DSIsNextToProblematicEB isNextToProblematicEB;
        isNextToProblematicEB.setChannelStatus(chStatus);
        isNextToProblematicEB.setStatusThreshold(0);

        DSIsNextToNextToProblematicEB isNextToNextToProblematicEB;
        isNextToNextToProblematicEB.setChannelStatus(chStatus);
        isNextToNextToProblematicEB.setStatusThreshold(0);

        DSIsNextToProblematicEE isNextToProblematicEE;
        isNextToProblematicEE.setChannelStatus(chStatus);

        DSHasChannelStatusEB hasChannelStatus;
        hasChannelStatus.setChannelStatus(chStatus);

        DSIsNextToBoundaryEB isNextToBoundaryEB;
        DSAll all;

        TFile * fout = new TFile("output_histos.root", "recreate");

        IC ic;

        TProfile * p_eta_tmp = new TProfile("p_eta_tmp", "p_eta_tmp", 171+120, -145.5, 145.5);
        IC::profileEta(ic, p_eta_tmp, all);
        fout->Write();
        fout->Close();
        return 0;

        ////IC::readCmscondXMLFile("MC_production/EcalIntercalibConstantsMC_2010_V3_Bon_mc.xml", ic);
        //IC::readTextFile("MC_production/EcalIntercalibConstants_V20120109_Electrons_etaScale.dat", ic);
        //IC::applyTwoCrystalEffect(ic);
        //IC::dumpXML(ic, "EcalIntercalibConstants_V20120109_Electrons_etaScale_BiXtal.xml", all);
        ////IC::dumpXML(ic, "EcalIntercalibConstants_V20120109_Electrons_etaScale.xml", all);

        //IC::readXMLFile("MC_production/IC_errors.xml", ic);
        //IC::dump(ic, "IC_errrors.dat", all);

        //IC::readTextFile("MC_production/EcalIntercalibConstants_V20120109_Electrons_etaScale_withErrors.dat", ic);
        //IC::applyTwoCrystalEffect(ic);
        //IC ic_smear;
        //IC::smear(ic, ic_smear);
        //IC::dump(ic_smear, "EcalIntercalibConstants_V20120109_Electrons_etaScale_withErrors_smeared.dat", all);

        //IC::readTextFile("MC_production/EcalIntercalibConstants_2010_V3_Bon_startup_mc.dat", ic);
        //IC::applyTwoCrystalEffect(ic);
        //IC::dumpXML(ic, "EcalIntercalibConstantsMC_2010_V3_Bon_mc_BiXtal.xml", all);

        IC::readXMLFile("MC_ideal_2011_V3.xml", ic);
        IC::applyTwoCrystalEffect(ic);
        IC::dumpXML(ic, "EcalIntercalibConstants_IDEAL.xml", all);

        return 0;
        //////////IC::readXMLFile("dump_test.xml", ic);
        //////////IC::dumpXML(ic, "dump_test_check.xml", all);

        //IC::readSimpleTextFile("ic_test.dat", ic);
        //IC::readSimpleTextFile("dump_GR_R_42_V21B.dat", ic);
        ////IC::readTextFile("interCalibConstants.combinedPi0Eta.run178003to180252.EcalBarrel_corrected.txt", ic);
        //IC::readTextFile("/afs/cern.ch/user/y/yangyong/public/InterCalibConstants/laserdata_20111122_158851_180363_EEfloatAlphav1/interCalibConstants.combinedPi0Eta.run160404to180252.EcalBarrel.txt", ic);
        IC::readTextFile("interCalibConstants.combinedPi0Eta.run160404to180252.EcalBarrel_corrected.txt", ic);

        //IC::readTextFile("recovery_new_prediction.dat", ic);
        //IC::readTextFile("recovery_prediction.dat", ic);
        TProfile * p_eta = new TProfile("p_eta", "p_eta", 171+120, -145.5, 145.5);
        IC::profileEta(ic, p_eta, all);
        IC::dump(ic, "pi0_coeff.dat", all);

        //And both(isBarrel, isEndcap);
        //printf("test logic: %f\n", IC::average(ic, both));

        ///////////DSAll aall;
        ///////////IC::paciocchiata(ic);
        ///////////IC::dump(ic, "eta_scale_tmp.dat", aall);
        ///////////TH2F * h2EE_p = new TH2F("h2EE_p", "h2EE_p", 101, -0.5, 100.5, 101, -0.5, 100.5);
        ///////////DSIsEndcapPlus isEndcapPlus;
        ///////////DSIsEndcapMinus isEndcapMinus;
        ///////////IC::constantMap(ic, h2EE_p, isEndcapPlus);
        ///////////h2EE_p->Draw("colz");
        ///////////gPad->Print("mapEE_plus.png");
        ///////////h2EE_p->Reset();
        ///////////IC::constantMap(ic, h2EE_p, isEndcapMinus);
        ///////////h2EE_p->Draw("colz");
        ///////////gPad->Print("mapEE_minus.png");
        ///////////return 0;
        printf("EB chStatus = 0 average: %f\n", IC::average(ic, hasChannelStatus));

        // fast check
        FILE * fd = fopen("nexttodead_eb_combinedic.txt", "r");
        int ieta, iphi, cnt = 0;
        float val = 0;
        while( fscanf(fd, "%d %d", &ieta, &iphi) != EOF) {
                val += *(ic.ic().find(EBDetId(ieta, iphi)));
                //printf("--> %d %d : %f\n", ieta, iphi, *(ic.ic().find(EBDetId(ieta, iphi))));
                ++cnt;
        }
        printf("%f %f %d\n", val / cnt, val, cnt);
        printf("EB next to dead average: %f\n", IC::average(ic, isNextToProblematicEB));
        printf("  (cnt = %d)\n", isNextToProblematicEB.cnt());
        printf("EB next to next to dead average: %f\n", IC::average(ic, isNextToNextToProblematicEB));
        printf("  (cnt = %d)\n", isNextToNextToProblematicEB.cnt());

        DSRandom isRand(0.024);
        isRand.setSeed(time(NULL));
        printf("EB for random channels (seed: %u): %f\n", isRand.seed(), IC::average(ic, isRand));


        TH2F * h2EB_tbd = new TH2F("h2EB_tbd", "h2EB_tbd", 361, -0.5, 360.5, 171, -85.5, 85.5);
        IC::constantMap(ic, h2EB_tbd, isNextToNextToProblematicEB);
        //IC::constantMap(ic, h2EB_tbd, isNextToProblematicEB);
        h2EB_tbd->Draw("colz");
        gPad->Print("mapEB_nextToNext.png");

        //IC::dumpXML(ic, "dump_test.xml", all);
        //return 0;

        //IC::readTextFile("interCalibConstants.combinedPi0Eta.run160404to180252.EcalBarrel_corrected.txt", ic);

        //ic.setEcalChannelStatus(&chStatus);
        TH1F * h = new TH1F("h", "h", 150, 0.5, 1.5);
        //constantDistribution(ic, h, isBarrel);
        IC::constantDistribution(ic, h, isEndcap);
        h->Draw();
        gPad->Print("distribution.png");

        TH2F * h2EB = new TH2F("h2EB", "h2EB", 361, -0.5, 360.5, 171, -85.5, 85.5);
        IC::constantMap(ic, h2EB, isNextToProblematicEB);
        h2EB->Draw("colz");
        gPad->Print("mapEB.png");

        TH2F * h2EE = new TH2F("h2EE", "h2EE", 101, -0.5, 100.5, 101, -0.5, 100.5);
        IC::constantMap(ic, h2EE, isNextToProblematicEE);
        h2EE->Draw("colz");
        gPad->Print("mapEE.png");

        IC ic_smeared;
        IC::smear(ic, 0.02, ic_smeared);

        IC ic_diff;
        IC::multiply(ic_smeared, -1, ic_smeared);
        IC::add(ic_smeared, ic, ic_diff);
        TH1F * h_diff = new TH1F("h_diff", "h_diff", 150, -0.5, 0.5);
        IC::constantDistribution(ic_diff, h_diff, isEndcap);

        IC ic_ratio;
        IC::reciprocal(ic, ic_ratio);
        IC::multiply(ic, ic_ratio, ic_ratio);
        TH1F * h_ratio = new TH1F("h_ratio", "h_ratio", 150, 0.5, 1.5);
        IC::constantDistribution(ic_ratio, h_ratio, all);

        TProfile * p_eta_EB = new TProfile("p_eta_EB", "p_eta_EB", 171, -85.5, 85.5);
        IC::profileEta(ic, p_eta_EB, isBarrel);
        TProfile * p_phi_EB = new TProfile("p_phi_EB", "p_phi_EB", 360, 0.5, 360.5);
        IC::profilePhi(ic, p_phi_EB, isBarrel);

        //TProfile * p_eta_EE = new TProfile("p_eta_EE", "p_eta_EE", 171, -85.5, 85.5);
        //profileEta(ic, p_eta_EE, isBarrel);
        TProfile * p_phi_EE = new TProfile("p_phi_EE", "p_phi_EE", 360, 0.5, 360.5);
        IC::profilePhi(ic, p_phi_EE, isEndcap);

        TProfile * p_SM = new TProfile("p_SM", "p_SM", 36, 0.5, 36.5);
        IC::profileSM(ic, p_SM, all);

        printf("overall average: %f\n", IC::average(ic, all));
        printf("EB average: %f\n", IC::average(ic, isBarrel));
        printf("EE average: %f\n", IC::average(ic, isEndcap));
        printf("EB next to borders average: %f\n", IC::average(ic, isNextToBoundaryEB));
        isNextToProblematicEB.reset();
        float av = IC::average(ic, isNextToProblematicEB);
        printf("EB next to dead average: %f (%d crystals)\n", av, isNextToProblematicEB.cnt());

        fout->Write();
        fout->Close();
        return 0;
}

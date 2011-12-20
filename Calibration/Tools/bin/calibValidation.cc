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
        IC ic;
        IC::readSimpleTextFile("ic_test.dat", ic);

        EcalChannelStatus chStatus;
        IC::readEcalChannelStatusFromTextFile("chStatus.dat");

        //ic.setEcalChannelStatus(&chStatus);

        TFile * fout = new TFile("output_histos.root", "recreate");
        TH1F * h = new TH1F("h", "h", 150, 0.5, 1.5);
        //constantDistribution(ic, h, isBarrel);
        IC::constantDistribution(ic, h, IC::isEndcap);
        h->Draw();
        gPad->Print("distribution.png");

        TH2F * h2EB = new TH2F("h2EB", "h2EB", 361, -0.5, 360.5, 171, -85.5, 85.5);
        IC::constantMap(ic, h2EB, IC::isNextToProblematicEB);
        h2EB->Draw("colz");
        gPad->Print("mapEB.png");

        TH2F * h2EE = new TH2F("h2EE", "h2EE", 101, -0.5, 100.5, 101, -0.5, 100.5);
        IC::constantMap(ic, h2EE, IC::isNextToProblematicEE);
        h2EE->Draw("colz");
        gPad->Print("mapEE.png");

        IC ic_smeared;
        IC::smear(ic, 0.02, ic_smeared);

        IC ic_diff;
        IC::multiply(ic_smeared, -1, ic_smeared);
        IC::add(ic_smeared, ic, ic_diff);
        TH1F * h_diff = new TH1F("h_diff", "h_diff", 150, -0.5, 0.5);
        IC::constantDistribution(ic_diff, h_diff, IC::isEndcap);

        IC ic_ratio;
        IC::reciprocal(ic, ic_ratio);
        IC::multiply(ic, ic_ratio, ic_ratio);
        TH1F * h_ratio = new TH1F("h_ratio", "h_ratio", 150, 0.5, 1.5);
        IC::constantDistribution(ic_ratio, h_ratio, IC::all);

        TProfile * p_eta_EB = new TProfile("p_eta_EB", "p_eta_EB", 171, -85.5, 85.5);
        IC::profileEta(ic, p_eta_EB, IC::isBarrel);
        TProfile * p_phi_EB = new TProfile("p_phi_EB", "p_phi_EB", 360, 0.5, 360.5);
        IC::profilePhi(ic, p_phi_EB, IC::isBarrel);

        //TProfile * p_eta_EE = new TProfile("p_eta_EE", "p_eta_EE", 171, -85.5, 85.5);
        //profileEta(ic, p_eta_EE, isBarrel);
        TProfile * p_phi_EE = new TProfile("p_phi_EE", "p_phi_EE", 360, 0.5, 360.5);
        IC::profilePhi(ic, p_phi_EE, IC::isEndcap);

        TProfile * p_SM = new TProfile("p_SM", "p_SM", 36, 0.5, 36.5);
        IC::profileSM(ic, p_SM, IC::all);

        printf("overall average: %f\n", IC::average(ic, IC::all));
        printf("EB average: %f\n", IC::average(ic, IC::isBarrel));
        printf("EE average: %f\n", IC::average(ic, IC::isEndcap));
        printf("EB borders average: %f\n", IC::average(ic, IC::isNextToBoundaryEB));
        printf("dead average: %f\n", IC::average(ic, IC::isNextToProblematicEB));

        fout->Write();
        fout->Close();
        return 0;
}

{
const TString titleEntriesBase = "Total Number of Entries: ";
const TString titleOffDiagonalBase = "Off Diagonal Entries: ";
TString titleEntries,  titleOffDiagonal;

const TString titleNumberOfMismatchesBase = "Number of Mismatches: ";
const TString titleNumberOfLegalMismatchesBase = "Number of Legal Mismatches: ";
TString titleNumberOfMismatches, titleNumberOfLegalMismatches;

//gStyle->TStyle::SetOptStat("emrruo");
gStyle->TStyle::SetOptStat("");

TCanvas cPt("cPt", "Pt");
comparePt.Draw();
TPaveText labelComparePt(0.72, 0.95,1.0, 1.0, "NDC"); 
TPaveText labelComparePtOffDiagonal(0.72, 0.90,1.0, 0.95, "NDC"); 
labelComparePt->SetBorderSize(1);
labelComparePtOffDiagonal->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += comparePtOffDiagonal.GetEntries();
labelComparePtOffDiagonal.AddText(titleOffDiagonal);
labelComparePtOffDiagonal.Draw();

TCanvas cPtOffDiagonal("cPtOffDiagonal", "Pt Off Diagonal");           
comparePtOffDiagonal.Draw();         

TCanvas cPtDifference("cPtDifference", "Pt Difference");
differencePt.SetFillColor(4);
differencePt.Draw();

TCanvas cPt_front("cPt_front", "Pt (Front)");
comparePt_front.Draw();         
TPaveText labelComparePtOffDiagonal_front(0.72, 0.95,1.0, 1.0, "NDC"); 
labelComparePtOffDiagonal_front->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += comparePtOffDiagonal_front.GetEntries();
labelComparePtOffDiagonal_front.AddText(titleOffDiagonal);
labelComparePtOffDiagonal_front.Draw();

TCanvas cPtOffDiagonal_front("cPtOffDiagonal_front", "Pt Off Diagonal (Front)");           
comparePtOffDiagonal_front.Draw();         

TCanvas cPtDifference_front("cPtDifference_front", "Pt Difference (Front)");
differencePt_front.SetFillColor(4);
differencePt_front.Draw();         

TCanvas cPt_rear("cPt_rear", "Pt (Rear)");            
comparePt_rear.Draw();         
TPaveText labelComparePtOffDiagonal_rear(0.72, 0.95,1.0, 1.0, "NDC"); 
labelComparePtOffDiagonal_rear->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += comparePtOffDiagonal_rear.GetEntries();
labelComparePtOffDiagonal_rear.AddText(titleOffDiagonal);
labelComparePtOffDiagonal_rear.Draw();

TCanvas cPtOffDiagonal_rear("cPtOffDiagonal_rear", "Pt Off Diagonal (Rear)");
comparePtOffDiagonal_rear.Draw();

TCanvas cPtDifference_rear("cPtDifference_rear", "Pt Difference (Rear)");
differencePt_rear.SetFillColor(4);
differencePt_rear.Draw();         


TCanvas cLocalPhi("cLocalPhi", "Local Phi");
compareLocalPhi.Draw();             
TPaveText labelCompareLocalPhiOffDiagonal(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareLocalPhiOffDiagonal->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareLocalPhiOffDiagonal.GetEntries();
labelCompareLocalPhiOffDiagonal.AddText(titleOffDiagonal);
labelCompareLocalPhiOffDiagonal.Draw();

TCanvas cLocalPhiOffDiagonal("cLocalPhiOffDiagonal", "Local Phi Off Diagonal");
compareLocalPhiOffDiagonal.Draw();             

TCanvas cLocalPhiDifference("cLocalPhiDifference", "Local Phi Differrence");
differenceLocalPhi.SetFillColor(4);
differenceLocalPhi.Draw();             

TCanvas cLocalPhi_phiLocal("cLocalPhi_phiLocal", "Local Phi (Phi Local Word)");
compareLocalPhi_phiLocal.Draw();             
TPaveText labelCompareLocalPhiOffDiagonal_phiLocal(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareLocalPhiOffDiagonal_phiLocal->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareLocalPhiOffDiagonal_phiLocal.GetEntries();
labelCompareLocalPhiOffDiagonal_phiLocal.AddText(titleOffDiagonal);
labelCompareLocalPhiOffDiagonal_phiLocal.Draw();

TCanvas cLocalPhiOffDiagonal_phiLocal("cLocalPhiOffDiagonal_phiLocal", "Local Phi Off Diagonal (Phi Local Word)");
compareLocalPhiOffDiagonal_phiLocal.Draw();             

TCanvas cLocalPhiDifference_phiLocal("cLocalPhiDifference_phiLocal", "Local Phi Difference (Phi Local Word)");
differenceLocalPhi_phiLocal.SetFillColor(4);
differenceLocalPhi_phiLocal.Draw();             

TCanvas cLocalPhi_phiBend("cLocalPhi_phiBend", "Local Phi (Phi Bend Word)");
compareLocalPhi_phiBend.Draw();             
TPaveText labelCompareLocalPhiOffDiagonal_phiBend(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareLocalPhiOffDiagonal_phiBend->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareLocalPhiOffDiagonal_phiBend.GetEntries();
labelCompareLocalPhiOffDiagonal_phiBend.AddText(titleOffDiagonal);
labelCompareLocalPhiOffDiagonal_phiBend.Draw();

TCanvas cLocalPhiOffDiagonal_phiBend("cLocalPhiOffDiagonal_phiBend", "Local Phi Off Diagonal (Phi Bend Word)");
compareLocalPhiOffDiagonal_phiBend.Draw();             

TCanvas cLocalPhiDifference_phiBend("cLocalPhiDifference_phiBend", "Local Phi Difference (Phi Bend Word)");
differenceLocalPhi_phiBend.SetFillColor(4);
differenceLocalPhi_phiBend.Draw();             


TCanvas cGlobalPhiME("cGlobalPhiME", "Global Phi ME");
compareGlobalPhiME.Draw();                 
TPaveText labelCompareGlobalPhiMEOffDiagonal(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareGlobalPhiMEOffDiagonal->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareGlobalPhiMEOffDiagonal.GetEntries();
labelCompareGlobalPhiMEOffDiagonal.AddText(titleOffDiagonal);
labelCompareGlobalPhiMEOffDiagonal.Draw();

TCanvas cGlobalPhiMEOffDiagonal("cGlobalPhiMEOffDiagonal", "Global Phi ME Off Diagonal");
compareGlobalPhiMEOffDiagonal.Draw();                 

TCanvas cGlobalPhiMEDifference("cGlobalPhiMEDifference", "Global Phi Difference ME");
differenceGlobalPhiME.SetFillColor(4);
differenceGlobalPhiME.Draw();                 


TCanvas cGlobalPhiMB("cGlobalPhiMB", "Global Phi MB");
compareGlobalPhiMB.Draw();                 
TPaveText labelCompareGlobalPhiMBOffDiagonal(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareGlobalPhiMBOffDiagonal->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareGlobalPhiMBOffDiagonal.GetEntries();
labelCompareGlobalPhiMBOffDiagonal.AddText(titleOffDiagonal);
labelCompareGlobalPhiMBOffDiagonal.Draw();

TCanvas cGlobalPhiMBOffDiagonal("cGlobalPhiMBOffDiagonal", "Global Phi MB Off Diagonal");
compareGlobalPhiMBOffDiagonal.Draw();                 

TCanvas cGlobalPhiMBDifference("cGlobalPhiMBDifference", "Global Phi MB Difference");
differenceGlobalPhiMB.SetFillColor(4);
differenceGlobalPhiMB.Draw();


TCanvas cGlobalEta("cGlobalEta", "Global Eta");       
compareGlobalEta.Draw();                   
TPaveText labelCompareGlobalEtaOffDiagonal(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareGlobalEtaOffDiagonal->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareGlobalEtaOffDiagonal.GetEntries();
labelCompareGlobalEtaOffDiagonal.AddText(titleOffDiagonal);
labelCompareGlobalEtaOffDiagonal.Draw();

TCanvas cGlobalEtaOffDiagonal("cGlobalEtaOffDiagonal", "Global Eta Off Diagonal");       
compareGlobalEtaOffDiagonal.Draw();                   

TCanvas cGlobalEtaDifference("cGlobalEtaDifference", "Global Eta Difference");       
differenceGlobalEta.SetFillColor(4);
differenceGlobalEta.Draw();                   

TCanvas cGlobalEta_etaGlobal("cGlobalEta_etaGlobal", "Global Eta (Eta Global Word)");       
compareGlobalEta_etaGlobal.Draw();                   
TPaveText labelCompareGlobalEtaOffDiagonal_etaGlobal(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareGlobalEtaOffDiagonal_etaGlobal->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareGlobalEtaOffDiagonal.GetEntries();
labelCompareGlobalEtaOffDiagonal_etaGlobal.AddText(titleOffDiagonal);
labelCompareGlobalEtaOffDiagonal_etaGlobal.Draw();

TCanvas cGlobalEtaOffDiagonal_etaGlobal("cGlobalEtaOffDiagonal_etaGlobal", "Global Eta Off Diagonal (Eta Global Word)");
compareGlobalEtaOffDiagonal_etaGlobal.Draw();                   

TCanvas cGlobalEtaDifference_etaGlobal("cGlobalEtaDifference_etaGlobal", "Global Eta Difference (Eta Global Word)");       
differenceGlobalEta_etaGlobal.SetFillColor(4);
differenceGlobalEta_etaGlobal.Draw();                   

TCanvas cGlobalEta_phiBend("cGlobalEta_phiBend", "Global Eta (Phi Bend Word)");       
compareGlobalEta_phiBend.Draw();                   
TPaveText labelCompareGlobalEtaOffDiagonal_phiBend(0.72, 0.95,1.0, 1.0, "NDC"); 
labelCompareGlobalEtaOffDiagonal_phiBend->SetBorderSize(1);
titleOffDiagonal = titleOffDiagonalBase;
titleOffDiagonal += compareGlobalEtaOffDiagonal_phiBend.GetEntries();
labelCompareGlobalEtaOffDiagonal_phiBend.AddText(titleOffDiagonal);
labelCompareGlobalEtaOffDiagonal_phiBend.Draw();

TCanvas cGlobalEtaOffDiagonal_phiBend("cGlobalEtaOffDiagonal_phiBend", "Global Eta Off Diagonal (Phi Bend Word)");       
compareGlobalEtaOffDiagonal_phiBend.Draw();                   

TCanvas cGlobalEtaDifference_phiBend("cGlobalEtaDifference_phiBend", "Global Eta Difference (Phi Bend Word)");       
differenceGlobalEta_phiBend.SetFillColor(4);
differenceGlobalEta_phiBend.Draw();                   


TCanvas cMismatchLocalPhiAddress("cMismatchLocalPhiAddress", "Mismatch Local Phi Address");
mismatchLocalPhiAddress.SetFillColor(4);  
mismatchLocalPhiAddress.Draw();  
TPaveText labelNumberOfLocalPhiMismatches(0.72, 0.95,1.0, 1.0, "NDC"); 
labelNumberOfLocalPhiMismatches.SetBorderSize(1);
TPaveText labelNumberOfLocalPhiLegalMismatches(0.72, 0.90, 1.0, 0.95, "NDC"); 
labelNumberOfLocalPhiLegalMismatches.SetBorderSize(1);
titleNumberOfMismatches = titleNumberOfMismatchesBase;
titleNumberOfMismatches += mismatchLocalPhiAddress.GetEntries();
titleNumberOfLegalMismatches = titleNumberOfLegalMismatchesBase;
titleNumberOfLegalMismatches += mismatchLocalPhiAddress_patternId.GetEntries();
labelNumberOfLocalPhiMismatches.AddText(titleNumberOfMismatches);
labelNumberOfLocalPhiLegalMismatches.AddText(titleNumberOfLegalMismatches);
labelNumberOfLocalPhiMismatches.Draw();
labelNumberOfLocalPhiLegalMismatches.Draw();


TCanvas cMismatchLocalPhiAddress_patternId("cMismatchLocalPhiAddress_patternId", "Mismatch Local Phi Address (Pattern ID Word)");
mismatchLocalPhiAddress_patternId.SetFillColor(4);  
mismatchLocalPhiAddress_patternId.Draw();

TCanvas cMismatchLocalPhiAddress_patternNumber("cMismatchLocalPhiAddress_patternNumber", "Mismatch Local Phi Address (Pattern Number Word)");
mismatchLocalPhiAddress_patternNumber.SetFillColor(4);  
mismatchLocalPhiAddress_patternNumber.Draw();  

TCanvas cMismatchLocalPhiAddress_quality("cMismatchLocalPhiAddress_quality", "Mismatch Local Phi Address (Quality Word)");
mismatchLocalPhiAddress_quality.SetFillColor(4);  
mismatchLocalPhiAddress_quality.Draw();  

TCanvas cMismatchLocalPhiAddress_leftRight("cMismatchLocalPhiAddress_leftRight", "Mismatch Local Phi Address (Left/Right Word)");
mismatchLocalPhiAddress_leftRight.SetFillColor(4);  
mismatchLocalPhiAddress_leftRight.Draw();  

TCanvas cMismatchLocalPhiAddress_spare("cMismatchLocalPhiAddress_spare", "Mismatch Local Phi Address (Spare Word)");
mismatchLocalPhiAddress_spare.SetFillColor(4);  
mismatchLocalPhiAddress_spare.Draw();  


TCanvas cMismatchGlobalEtaAddress("cMismatchGlobalEtaAddress", "Mismatch Global Eta Address");
mismatchGlobalEtaAddress.SetFillColor(4);  
mismatchGlobalEtaAddress.Draw();  
TPaveText labelNumberOfGlobalEtaMismatches(0.72, 0.95,1.0, 1.0, "NDC"); 
labelNumberOfGlobalEtaMismatches.SetBorderSize(1);
TPaveText labelNumberOfGlobalEtaLegalMismatches(0.72, 0.90, 1.0, 0.95, "NDC"); 
labelNumberOfGlobalEtaLegalMismatches.SetBorderSize(1);
titleNumberOfMismatches = titleNumberOfMismatchesBase;
titleNumberOfMismatches += mismatchGlobalEtaAddress.GetEntries();
titleNumberOfLegalMismatches = titleNumberOfLegalMismatchesBase;
titleNumberOfLegalMismatches += mismatchGlobalEtaAddress_cscId.GetEntries();
labelNumberOfGlobalEtaMismatches.AddText(titleNumberOfMismatches);
labelNumberOfGlobalEtaLegalMismatches.AddText(titleNumberOfLegalMismatches);
labelNumberOfGlobalEtaMismatches.Draw();
labelNumberOfGlobalEtaLegalMismatches.Draw();

TCanvas cMismatchGlobalEtaAddress_phiBendLocal("cMismatchGlobalEtaAddress_phiBendLocal", "Mismatch Global Eta Address (Phi Bend Local Word)");
mismatchGlobalEtaAddress_phiBendLocal.SetFillColor(4);  
mismatchGlobalEtaAddress_phiBendLocal.Draw();  

TCanvas cMismatchGlobalEtaAddress_phiLocal("cMismatchGlobalEtaAddress_phiLocal", "Mismatch Global Eta Address (Phi Local Word)");
mismatchGlobalEtaAddress_phiLocal.SetFillColor(4);  
mismatchGlobalEtaAddress_phiLocal.Draw();  

TCanvas cMismatchGlobalEtaAddress_wireGroup("cMismatchGlobalEtaAddress_wireGroup", "Mismatch Global Eta Address (Wire Group Word)");
mismatchGlobalEtaAddress_wireGroup.SetFillColor(4);  
mismatchGlobalEtaAddress_wireGroup.Draw();  

TCanvas cMismatchGlobalEtaAddress_cscId("cMismatchGlobalEtaAddress_cscId", "Mismatch Global Eta Address (CSC ID Word)");
mismatchGlobalEtaAddress_cscId.SetFillColor(4);  
mismatchGlobalEtaAddress_cscId.Draw();  


TCanvas cMismatchGlobalPhiMEAddress("cMismatchGlobalPhiMEAddress", "Mismatch Global Phi ME Address");
mismatchGlobalPhiMEAddress.SetFillColor(4);  
mismatchGlobalPhiMEAddress.Draw();  
TPaveText labelNumberOfGlobalPhiMEMismatches(0.72, 0.95,1.0, 1.0, "NDC"); 
labelNumberOfGlobalPhiMEMismatches.SetBorderSize(1);
TPaveText labelNumberOfGlobalPhiMELegalMismatches(0.72, 0.90, 1.0, 0.95, "NDC"); 
labelNumberOfGlobalPhiMELegalMismatches.SetBorderSize(1);
titleNumberOfMismatches = titleNumberOfMismatchesBase;
titleNumberOfMismatches += mismatchGlobalPhiMEAddress.GetEntries();
titleNumberOfLegalMismatches = titleNumberOfLegalMismatchesBase;
titleNumberOfLegalMismatches += mismatchGlobalPhiMEAddress_cscId.GetEntries();
labelNumberOfGlobalPhiMEMismatches.AddText(titleNumberOfMismatches);
labelNumberOfGlobalPhiMELegalMismatches.AddText(titleNumberOfLegalMismatches);
labelNumberOfGlobalPhiMEMismatches.Draw();
labelNumberOfGlobalPhiMELegalMismatches.Draw();

TCanvas cMismatchGlobalPhiMEAddress_phiLocal("cMismatchGlobalPhiMEAddress_phiLocal", "Mismatch Global Phi ME Address (Phi Local Word)");
mismatchGlobalPhiMEAddress_phiLocal.SetFillColor(4);  
mismatchGlobalPhiMEAddress_phiLocal.Draw();  

TCanvas cMismatchGlobalPhiMEAddress_wireGroup("cMismatchGlobalPhiMEAddress_wireGroup", "Mismatch Global Phi ME Address (Wire Group Word)");
mismatchGlobalPhiMEAddress_wireGroup.SetFillColor(4);  
mismatchGlobalPhiMEAddress_wireGroup.Draw();  

TCanvas cMismatchGlobalPhiMEAddress_cscId("cMismatchGlobalPhiMEAddress_cscId", "Mismatch Global Phi ME Address (CSC ID Word)");
mismatchGlobalPhiMEAddress_cscId.SetFillColor(4);  
mismatchGlobalPhiMEAddress_cscId.Draw();  


TCanvas cMismatchGlobalPhiMBAddress("cMismatchGlobalPhiMBAddress", "Mismatch Global Phi MB Address");
mismatchGlobalPhiMBAddress.SetFillColor(4);  
mismatchGlobalPhiMBAddress.Draw();  
TPaveText labelNumberOfGlobalPhiMBMismatches(0.72, 0.95,1.0, 1.0, "NDC"); 
labelNumberOfGlobalPhiMBMismatches.SetBorderSize(1);
TPaveText labelNumberOfGlobalPhiMBLegalMismatches(0.72, 0.90, 1.0, 0.95, "NDC"); 
labelNumberOfGlobalPhiMBLegalMismatches.SetBorderSize(1);
titleNumberOfMismatches = titleNumberOfMismatchesBase;
titleNumberOfMismatches += mismatchGlobalPhiMBAddress.GetEntries();
titleNumberOfLegalMismatches = titleNumberOfLegalMismatchesBase;
titleNumberOfLegalMismatches += mismatchGlobalPhiMBAddress_cscId.GetEntries();
labelNumberOfGlobalPhiMBMismatches.AddText(titleNumberOfMismatches);
labelNumberOfGlobalPhiMBLegalMismatches.AddText(titleNumberOfLegalMismatches);
labelNumberOfGlobalPhiMBMismatches.Draw();
labelNumberOfGlobalPhiMBLegalMismatches.Draw();

TCanvas cMismatchGlobalPhiMBAddress_phiLocal("cMismatchGlobalPhiMBAddress_phiLocal", "Mismatch Global Phi MB Address (Phi Local Word)");
mismatchGlobalPhiMBAddress_phiLocal.SetFillColor(4);  
mismatchGlobalPhiMBAddress_phiLocal.Draw();  

TCanvas cMismatchGlobalPhiMBAddress_wireGroup("cMismatchGlobalPhiMBAddress_wireGroup", "Mismatch Global Phi MB Address (Wire Group Word)");
mismatchGlobalPhiMBAddress_wireGroup.SetFillColor(4);  
mismatchGlobalPhiMBAddress_wireGroup.Draw();  

TCanvas cMismatchGlobalPhiMBAddress_cscId("cMismatchGlobalPhiMBAddress_cscId", "Mismatch Global Phi MB Address (CSC ID Word)");
mismatchGlobalPhiMBAddress_cscId.SetFillColor(4);  
mismatchGlobalPhiMBAddress_cscId.Draw();  


TCanvas cMismatchPtAddress("cMismatchPtAddress", "Mismatch Pt Address");
mismatchPtAddress.SetFillColor(4);  
mismatchPtAddress.Draw();  
TPaveText labelNumberOfPtMismatches(0.72, 0.95,1.0, 1.0, "NDC"); 
labelNumberOfPtMismatches.SetBorderSize(1);
TPaveText labelNumberOfPtLegalMismatches(0.72, 0.90, 1.0, 0.95, "NDC"); 
labelNumberOfPtLegalMismatches.SetBorderSize(1);
titleNumberOfMismatches = titleNumberOfMismatchesBase;
titleNumberOfMismatches += mismatchPtAddress.GetEntries();
//titleNumberOfLegalMismatches = titleNumberOfLegalMismatchesBase;
//titleNumberOfLegalMismatches += mismatchPtAddress_eta.GetEntries();
titleNumberOfLegalMismatches = "Currently no test for legal Pt Address";
labelNumberOfPtMismatches.AddText(titleNumberOfMismatches);
labelNumberOfPtLegalMismatches.AddText(titleNumberOfLegalMismatches);
labelNumberOfPtMismatches.Draw();
labelNumberOfPtLegalMismatches.Draw();

TCanvas cMismatchPtAddress_delta12phi("cMismatchPtAddress_delta12phi", "Mismatch Pt Address (Delta12Phi Word)");
mismatchPtAddress_delta12phi.SetFillColor(4);  
mismatchPtAddress_delta12phi.Draw();  

TCanvas cMismatchPtAddress_delta23phi("cMismatchPtAddress_delta23phi", "Mismatch Pt Address (Delta23Phi Word)");
mismatchPtAddress_delta23phi.SetFillColor(4);  
mismatchPtAddress_delta23phi.Draw();  

TCanvas cMismatchPtAddress_deltaPhi("cMismatchPtAddress_deltaPhi", "Mismatch Pt Address (DeltaPhi Word)");
mismatchPtAddress_deltaPhi.SetFillColor(4);  
mismatchPtAddress_deltaPhi.Draw();  

TCanvas cMismatchPtAddress_eta("cMismatchPtAddress_eta", "Mismatch Pt Address (Eta Word)");
mismatchPtAddress_eta.SetFillColor(4);  
mismatchPtAddress_eta.Draw();  

TCanvas cMismatchPtAddress_mode("cMismatchPtAddress_mode", "Mismatch Pt Address (Mode Word)");
mismatchPtAddress_mode.SetFillColor(4);  
mismatchPtAddress_mode.Draw();  

TCanvas cMismatchPtAddress_sign("cMismatchPtAddress_sign", "Mismatch Pt Address (Sign Word)");
mismatchPtAddress_sign.SetFillColor(4);  
mismatchPtAddress_sign.Draw();  


TCanvas cInputVsOutputLocalPhi_1("cInputVsOutputLocalPhi_1", "Input vs. Output Local Phi 1");
InputVsOutputLocalPhi_1.Draw();  

TCanvas cInputVsOutputGlobalEta_1("cInputVsOutputGlobalEta_1", "Input vs. Output Global Eta 1");
InputVsOutputGlobalEta_1.Draw();  

TCanvas cInputVsOutputGlobalPhiME_1("cInputVsOutputGlobalPhiME_1", "Input vs. Output Global Phi ME 1");
InputVsOutputGlobalPhiME_1.Draw();  

TCanvas cInputVsOutputGlobalPhiMB_1("cInputVsOutputGlobalPhiMB_1", "Input vs. Output Global Phi MB 1");
InputVsOutputGlobalPhiMB_1.Draw();  

TCanvas cInputVsOutputPt_1("cInputVsOutputPt_1", "Input vs. Output Pt 1");
InputVsOutputPt_1.Draw();  


TCanvas cInputVsOutputLocalPhi_2("cInputVsOutputLocalPhi_2", "Input vs. Output Local Phi 2");
InputVsOutputLocalPhi_2.Draw();  

TCanvas cInputVsOutputGlobalEta_2("cInputVsOutputGlobalEta_2", "Input vs. Output Global Eta 2");
InputVsOutputGlobalEta_2.Draw();  

TCanvas cInputVsOutputGlobalPhiME_2("cInputVsOutputGlobalPhiME_2", "Input vs. Output Global Phi ME 2");
InputVsOutputGlobalPhiME_2.Draw();  

TCanvas cInputVsOutputGlobalPhiMB_2("cInputVsOutputGlobalPhiMB_2", "Input vs. Output Global Phi MB 2");
InputVsOutputGlobalPhiMB_2.Draw();  

TCanvas cInputVsOutputPt_2("cInputVsOutputPt_2", "InputVsOutputPt 2");
InputVsOutputPt_2.Draw();  

}

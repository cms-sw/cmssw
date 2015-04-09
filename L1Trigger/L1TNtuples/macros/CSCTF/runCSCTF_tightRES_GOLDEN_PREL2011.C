{
gROOT->ProcessLine(".x ../initL1Analysis.C");
gROOT->ProcessLine(".L ../Style.C");
gROOT->ProcessLine("setTDRStyle()");
gROOT->ProcessLine("gROOT->ForceStyle()");
gROOT->ProcessLine(".L L1CSCTF_tightRES_GOLDEN_PREL2011.C++");
L1CSCTF_tightRES_GOLDEN_PREL2011 t;
//L1CSCTF_RES t("/data/raid5/kropiv/collision/test_new01.root");

//t.OpenWithList("list_PromptReconew_v5v6v1B.txt");
//t.OpenWithList("list_PromptReconewPTLUT_v5v6v1B.txt");
//t.OpenWithList("list_PromptReconewPTLUT230112_2a_v6v1B.txt");
//t.OpenWithList("list_PromptReconew_2a_v6v1B.txt");

//t.OpenWithList("list_PromptRecoPTLUT_v3.txt");
//t.OpenWithList("list_PromptRecoPTLUT_v5.txt");
//t.OpenWithList("list_PromptRecoPTLUT_v6part.txt");
//t.OpenWithList("list_PromptRecoPTLUT_v7part.txt");
//t.OpenWithList("list_PromptRecoPTLUT_v8.txt");
//t.OpenWithList("list_PromptRecoPTLUT_v9.txt");
//t.OpenWithList("list_PromptRecoPTLUT_v6a_part.txt");
//t.OpenWithList("list_PromptRecoPTLUT_v6b_part.txt");
//t.OpenWithList("list_PromptRecoPTLUT2011.txt");
//t.OpenWithList("list_PromptRecoPTLUT2011_1b.txt");
//t.OpenWithList("list_PromptRecoPTLUT2011emu.txt");
//t.OpenWithList("list_PromptReco_V02_01_02.txt");
//t.OpenWithList("list_PromptReco2012v1_DCSjson.txt");
t.OpenWithList("list_PromptReco2012APTLUT32_DCSjson.txt");

//set up PtCut = 10 GeV at program:
//t.OpenWithList("list_PromptRecoMu10new_1B.txt");
//t.OpenWithList("list_PromptRecoMu10newPTLUT_1B.txt");

//t.run(10000);
t.run(-1);
}

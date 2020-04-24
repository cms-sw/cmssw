{

gROOT->ProcessLine(".L tdrstyle.C");
setTDRStyle();
gStyle->SetErrorX(0.5);




//----------------------------------------------------------------------------------------------------------------------------



gROOT->ProcessLine(".L DrawIteration.C");


gROOT->ProcessLine("DrawIteration drawIteration1(14)");
drawIteration1.yAxisFixed(true);

//drawIteration1.drawIteration();
drawIteration1.drawIteration(1,8);
drawIteration1.drawIteration(9,14);
drawIteration1.drawIteration(15,26);
drawIteration1.drawIteration(27,38);
drawIteration1.drawIteration(39,48);
drawIteration1.drawIteration(49,68);


//gROOT->ProcessLine("DrawIteration drawIteration2(15)");
//drawIteration2.yAxisFixed(true);


//~ drawIteration2.drawIteration(1,8);
//~ drawIteration2.drawIteration(9,14);
//~ drawIteration2.drawIteration(15,26);
//~ drawIteration2.drawIteration(27,38);
//~ drawIteration2.drawIteration(39,48);
//~ drawIteration2.drawIteration(49,68);


gStyle->SetPadLeftMargin(0.15);
gStyle->SetPadRightMargin(0.10);
gStyle->SetTitleOffset(1.0,"Y");


//drawIteration1.addSystematics();
//drawIteration1.addCmsText("CMS Preliminary");
drawIteration1.drawResult();

//drawIteration2.addSystematics();
//drawIteration2.addCmsText("CMS Preliminary");
//drawIteration2.drawResult();

gROOT->ProcessLine(".q");


}

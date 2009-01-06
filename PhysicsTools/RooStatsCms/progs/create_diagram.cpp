#include <iostream>

#include "RooGlobalFunc.h"

#include "RscCombinedModel.h"


#include "TString.h"

int main (int argc, char** argv) {

if (argc!=3){
    std::cout << "Usage:\n"
              << argv[0] << " card_name combined_model_name\n";
    return 1;
    }

TString card_name=argv[1];
TString model_name=argv[2];

RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;

RscAbsPdfBuilder::setDataCard(card_name.Data());
RscCombinedModel combo(model_name.Data());

RooAbsPdf* sb_pdf=combo.getPdf();

RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

TString image_name=model_name+"_diagram.png";
TString dot_name=model_name+"_diagram.dot";

sb_pdf->graphVizTree(dot_name.Data());

TString dot_command="dot -Tpng "+dot_name+" -o "+image_name;

if (system(dot_command)!=0)
    std::cout << "dot program needed to produce plot!\n";

TString display_progs[]={"display ","gwenview ","firefox "};

bool displayed=false;
for (int i=0;i<3;++i){
    if (system((display_progs[i]+image_name).Data())==0)
        return 0;
    }

std::cout << "One of the programs needed to display plot:\n";
for (int i=0;i<3;++i)
    std::cout << " - " << display_progs[i]<< std::endl;

return 1;
}

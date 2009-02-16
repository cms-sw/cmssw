#include <iostream>
#include <fstream>
#include <sstream>


#include "RooGlobalFunc.h"

#include "RscCombinedModel.h"

#include "TString.h"
#include "TObjString.h"

int create_mainpage(const char* name, RooAbsPdf* pdf,RooArgList vars);
int create_page(const char* name, RooAbsArg* pdf, bool first_call);
int create_varspage(const char* name, RooAbsPdf* sb_model,bool first_call);


TObjArray ARR;

int main (int argc, char** argv) {

if (argc!=3){
    std::cout << "Usage:\n"
              << argv[0] << " card_name combined_model_name\n";
    return 1;
    }

TString card_name=argv[1];
TString model_name=argv[2];


// create the model
RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;

RscAbsPdfBuilder::setDataCard(card_name.Data());
// RscTotModel tot(model_name.Data());
// tot.getPdf();

RscCombinedModel combo(model_name.Data());

RooAbsPdf* sb_model=combo.getPdf();

RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

//TUnixSystem sys;

// make docu directory
//sys.MakeDirectory(model_name.Data());
system(("mkdir " + model_name).Data());

// mainpage
RooArgList vars(combo.getVars());
create_mainpage(model_name.Data(),sb_model,vars);

RooArgList l(*sb_model->getComponents());

for (int i=0;i<l.getSize();++i){
    RooAbsArg* obj = &(l[i]);
    create_page(model_name.Data(),obj,true);
    }

std::cout << "Creating pages for the parameters...\n";

RooArgList pars (*sb_model->getParameters(vars));
for (int i=0;i<pars.getSize();++i){
    RooAbsArg* obj = &(pars[i]);
    create_page(model_name.Data(),obj,true);
    }


create_varspage(model_name.Data(),sb_model,true);


// now put the link among the variables

system (("ls "+model_name+"/*.html > "+model_name+"/html_list.txt").Data());

string line;
ifstream myfile ((model_name+"/html_list.txt").Data());
if (myfile.is_open()){
    while (! myfile.eof() ){
        getline (myfile,line);
        cout << line << endl;
        TObjString* s = new TObjString(line.c_str());
        ARR.Add(s);
        }
    myfile.close();
    }
else 
    std::cerr << "Unable to open file"; 


for (int i=0;i<l.getSize();++i){
    RooAbsArg* obj = &(l[i]);
    create_page(model_name.Data(),obj,false);
    }


std::cout << "Creating pages for the parameters...\n";

for (int i=0;i<pars.getSize();++i){
    RooAbsArg* obj = &(pars[i]);
    create_page(model_name.Data(),obj,false);
    }

create_varspage(model_name.Data(),sb_model,false);

std::cout << "WebSite created in directory " << model_name.Data() << " \n";

ARR.Clear();

return 0;

}
/*----------------------------------------------------------------------------*/
int create_varspage(const char* name, RooAbsPdf* sb_model, bool first_call){

    TString model_name(name);
    TString ofilename(model_name);
    ofilename+="/variables_";
    ofilename+=model_name;
    ofilename+=".html";
    ofstream out(ofilename.Data());

    out << "<html>"
        << "<body>"
        << "<h1 align=\"center\"> Variables summary for: " << model_name.Data()<< "</h1><br><br>"
        << "<ul>";

    // the print statement
    std::stringstream sstream;
    std::streambuf* cout_sbuf = std::cout.rdbuf(); // save original sbuf
    std::cout.rdbuf(sstream.rdbuf()); // redirect 'cout' to a 'fout'
    sb_model->getVariables()->Print("v");
    std::cout.rdbuf(cout_sbuf); // restore the original stream buffer

    TString text(sstream.str().c_str());

    if (not first_call){
        TIterator* iter = ARR.MakeIterator();
        TObjString* ostr;
        while ((ostr = (TObjString*) (iter->Next()))!=NULL){
            TString thename(ostr->GetString().Data());
            thename.ReplaceAll(model_name+"/","");
            for (int i=0;i<5;i++){
                thename.Chop();
                //std::cout << thename.Data() << std::endl;
                }
            if (thename=="") continue;
            //std::cout << "HTML file " << (ostr->GetString()).Data() << std::endl;
            std::cout << "Name: " << thename.Data() << std::endl;
            if (text.Contains(thename)) {
                std::cout << "Contains " << thename.Data() << std::endl;
                TString replacement="<a href=\"";
                replacement+=thename+".html\">"+thename+"</a>";
                std::cout << "Replacing " << thename << " with " << replacement << std::endl;
                TString spaces="";
                for (int i=0;i<60;++i){
                    spaces+=" ";
                    text.ReplaceAll("::"+spaces+thename+" ","::"+replacement+" ");
                    }
                text.ReplaceAll(") "+thename+"<",") "+replacement+"<");
                text.ReplaceAll("> "+thename+"<","> "+replacement+"<");
                text.ReplaceAll("\""+thename+"\"","\""+replacement+"\"");
                }
            }
        }

    TObjArray* lines= text.Tokenize("\n");
    TIterator* iter = lines->MakeIterator();
    TObjString* ostr;
    bool first=true;
    while ((ostr = (TObjString*) (iter->Next()))!=NULL){
        if (first){
            first=false;
            continue;
            }
        TString line(ostr->GetString().Data());
        //out << "<li>"<<line.Data()<<"</li>";
        out <<line.Data()<<"<br>";
        }
    out.close();

    std::cout << "Vars file: " << ofilename.Data() << std::endl;
    return 0;
    }

/*----------------------------------------------------------------------------*/

int create_mainpage(const char* name, RooAbsPdf* sb_model,RooArgList vars){
    std::cout << "create_mainpage\n";
    TString ofilename(name);
    ofilename+="/index_";
    ofilename+=name;
    ofilename+=".html";
    ofstream out(ofilename.Data());
    out << "<html>"
        << "<body>"
        << "<h1 align=\"center\"> Combined model: " << name << "</h1><br><br>"
        << "Click <a href=\"variables_"<< name <<".html\">here</a> for the variables summary.<br>"
        << "Components of the model:<br><ul>";

    RooArgList l(*sb_model->getComponents());
    RooAbsArg* obj;
    TString o_name;
    for (int i=0;i<l.getSize();++i){
        obj = &(l[i]);
        TString o_name(obj->GetName());
        out << "<li> <a href=\""<< o_name.Data() << ".html\"> " << o_name.Data() << "</a></li>";
        }

    out << "</ul><br>";

    out << "Parameters:<ul>";
    RooArgList pars (*sb_model->getParameters(vars));
    RooAbsArg* par;
    for (int j=0;j<pars.getSize();++j){
        par = &(pars[j]);
        o_name=par->GetName();
        out << "<li> <a href=\""<< o_name.Data() << ".html\"> " << o_name.Data() << "</a></li>";
        }
    out << "</ul><br>";


    out << "Detailed version:"
        << "<ul>";

    for (int i=0;i<l.getSize();++i){
        obj = &(l[i]);
        o_name=obj->GetName();
        out << "<li> <a href=\""<< o_name.Data() << ".html\"> " << o_name.Data() << "</a></li>";

        out << "<ul>";
        RooArgList pars2(*obj->getParameters(vars));
        for (int j=0;j<pars2.getSize();++j){
            par = &(pars2[j]);
            o_name=par->GetName();
            out << "<li> <a href=\""<< o_name.Data() << ".html\"> " << o_name.Data() << "</a></li>";
            }
        out << "</ul><br>";
        }

    out << "</ul><br>"
        << "</body>"
        << "</html>";
    out.close();
    return 1;
    }

/*----------------------------------------------------------------------------*/

int create_page(const char* name, RooAbsArg* obj, bool first_call){
    std::cout << "create_page\n";

    TString model_name(name);
    TString ofilename(name);
    ofilename+="/";
    ofilename+=obj->GetName();
    ofilename+=".html";
    std::ofstream   out(ofilename.Data());

    out << "<html>"
        << "<body>"
        << "<h1 align=\"center\">" << obj->ClassName() << " component: " << obj->GetName() << "</h1><br><br>";

    // the image

    TString obj_name(obj->GetName());
    TString image_name=model_name+"/"+obj_name+"_diagram.png";
    TString dot_name=model_name+"/"+obj_name+"_diagram.dot";
    if (first_call){
        obj->graphVizTree(dot_name.Data());
        TString dot_command="dot -Tpng "+dot_name+" -o "+image_name;
        system(dot_command.Data());
        system(("convert -scale 50% "+image_name+" "+model_name+"/"+obj_name+"_diagram_thumb.png").Data());
        system(("rm -f "+dot_name).Data());
        }
    out << "<a href=\"" << (obj_name+"_diagram.png").Data() << "\" target=\"blank\">"
        << "<img src=\"" << (obj_name+"_diagram_thumb.png").Data() << "\" align=\"center\" border=0>"
        << "</a><br><font size=\"-1\"><em>Click on the diagram for the full size image.</em></font><br><br>";


    // the print statement
    std::stringstream sstream;
    std::streambuf* cout_sbuf = std::cout.rdbuf(); // save original sbuf
    std::cout.rdbuf(sstream.rdbuf()); // redirect 'cout' to a 'fout'
    obj->Print("v");
    std::cout.rdbuf(cout_sbuf); // restore the original stream buffer

    TString objinfo(sstream.str().c_str());

    objinfo.ReplaceAll("\n","<br>");

    // replace the names with hyperlinks among pages..
    if (not first_call){
        TIterator* iter = ARR.MakeIterator();
        TObjString* ostr;
        while ((ostr = (TObjString*) (iter->Next()))!=NULL){
            TString thename(ostr->GetString().Data());
            thename.ReplaceAll(model_name+"/","");
            for (int i=0;i<5;i++){
                thename.Chop();
                //std::cout << thename.Data() << std::endl;
                }
            if (thename=="") continue;
            //std::cout << "HTML file " << (ostr->GetString()).Data() << std::endl;
            std::cout << "Name: " << thename.Data() << std::endl;
            if (objinfo.Contains(thename)) {
//                 if (objinfo.Contains(thename)){
                std::cout << "Contains " << thename.Data() << std::endl;
                TString replacement="<a href=\"";
                replacement+=thename+".html\">"+thename+"</a>";
                std::cout << "Replacing " << thename << " with " << replacement << std::endl;
                objinfo.ReplaceAll("::"+thename+" ","::"+replacement+" ");
                objinfo.ReplaceAll(") "+thename+"<",") "+replacement+"<");
                objinfo.ReplaceAll("> "+thename+"<","> "+replacement+"<");
                objinfo.ReplaceAll("\""+thename+"\"","\""+replacement+"\"");
                }
            }
    std::cout << objinfo.Data();
        }

    out << objinfo.Data()
        << "</body>"
        << "</html>";

    out.close();
 
    return 1;
    }

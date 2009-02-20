// @(#)root/hist:$Id: PLScanPlot.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include <map>

#include "TAxis.h"
#include "TMath.h"

#include "PhysicsTools/RooStatsCms/interface/PLScanPlot.h"

#define TEXT_SIZE  0.035

/*----------------------------------------------------------------------------*/
PLScanPlot::PLScanPlot(const char* name,
                       const  char* title,
                       const char* scanned_var,
                       std::vector<double> x_points,
                       std::vector<double> y_points,
                       double val_at_min,
                       int float_digits,
                       bool verbosity)
    :StatisticalPlot(name,title,verbosity),
    m_cl_lines_num(0),
    m_cl_lines_tags_num(0),
    m_fc_graphs_num(0),
    m_float_digits(float_digits),
    m_val_at_min(val_at_min),
    m_minimum_tag(NULL){

    strcpy(m_scanned_var,scanned_var);

    assert(x_points.size()==y_points.size());

    // Prepare a map with the xy coordinates of the scan as content
    std::map<double,double> scan_content;

    for (int point=0;point<x_points.size();++point)
        scan_content[x_points[point]]=y_points[point];
    scan_content[val_at_min]=0.;

    m_scan_graph = new TGraph (x_points.size()+1);
    m_scan_graph->SetName("scan_graph");

    // Fill the graph
    int i=0;
    std::map<double,double>::iterator iter;
    for( iter = scan_content.begin();
         iter != scan_content.end();
         iter++ )
        m_scan_graph->SetPoint(i++,
                               iter->first,
                               iter->second);

    m_scan_graph->GetYaxis()->SetRangeUser(-2., 
                                           m_scan_graph->GetYaxis()->GetXmax());

    m_scan_graph->SetTitle(title);
    m_scan_graph->GetXaxis()->SetTitle(scanned_var);
    m_scan_graph->GetYaxis()->SetTitle("#Delta Nll");
    m_scan_graph->SetMarkerStyle(8);
    m_scan_graph->SetMarkerColor(kRed);
    m_scan_graph->SetMarkerSize(1);
    m_scan_graph->SetLineColor(kRed);

    // The minimum point of the scan
    m_min_point_graph = new TGraph(0);
    m_min_point_graph->SetName("min_point_graph");
    m_min_point_graph->SetPoint(0,val_at_min,0.);
    m_min_point_graph->SetMarkerStyle(8);
    m_min_point_graph->SetMarkerColor(kGreen);
    m_min_point_graph->SetMarkerSize(1);

    // The zero Line
    double left_margin=m_scan_graph->GetXaxis()->GetXmin();
    double right_margin=m_scan_graph->GetXaxis()->GetXmax();

    m_zero_line=new TLine(left_margin, 0,
                          right_margin, 0);
    }

/*----------------------------------------------------------------------------*/

int PLScanPlot::addCLline(double deltaNLL,
                          double CL,
                          double limit){

    if (m_cl_lines_num>=MAX_CL_LINES){
        std::cout << "INFO: Impossible to add another line.Limit reached.\n";
        return -1;
        }

    double left_margin=m_scan_graph->GetXaxis()->GetXmin();
    double right_margin=m_scan_graph->GetXaxis()->GetXmax();

    //double deltaNLL= m_getDeltaNLLfromCL(CL);

    // The line

    m_cl_lines[m_cl_lines_num]=new TLine(left_margin, deltaNLL,
                                         right_margin, deltaNLL);

    m_cl_lines[m_cl_lines_num]->SetLineWidth(2);
    m_cl_lines[m_cl_lines_num]->SetLineColor(m_cl_lines_num+3);

    ++m_cl_lines_num;

    // The line tag

    // two digits after the point
    double limit_f= double(int (limit*pow(10,m_float_digits)))/pow(10,m_float_digits);

    double tag_offset=0.15;

    TString tag_text="Limit ";
    tag_text+=limit_f;
    tag_text+=" at ";
    tag_text+=CL*100;
    tag_text+="% CL";

    m_cl_lines_tags[m_cl_lines_tags_num]=new TText(left_margin+tag_offset, 
                                              deltaNLL+tag_offset,
                                              tag_text.Data());

    m_cl_lines_tags[m_cl_lines_tags_num]->SetTextSize(TEXT_SIZE);
    m_cl_lines_tags[m_cl_lines_tags_num]->SetTextColor(m_cl_lines_num+3);

    ++m_cl_lines_tags_num;

    //getCanvas()->Update();

    }

/*----------------------------------------------------------------------------*/

int PLScanPlot::addCLline(double deltaNLL,
                          double CL,
                          double Llimit,
                          double Ulimit){

    if (m_cl_lines_num>=MAX_CL_LINES){
        std::cout << "INFO: Impossible to add another line.Limit reached.\n";
        return -1;
        }

    double left_margin=m_scan_graph->GetXaxis()->GetXmin();
    double right_margin=m_scan_graph->GetXaxis()->GetXmax();

    //double deltaNLL= m_getDeltaNLLfromCL(CL);

    // The line

    m_cl_lines[m_cl_lines_num]=new TLine(left_margin, deltaNLL,
                                         right_margin, deltaNLL);

    m_cl_lines[m_cl_lines_num]->SetLineWidth(2);
    m_cl_lines[m_cl_lines_num]->SetLineColor(m_cl_lines_num+3);

    ++m_cl_lines_num;

    // The line tag

    double tag_offset=0.15;

    // two digits after the point
    double Llimit_f= double(int (Llimit*pow(10,m_float_digits)))/pow(10,m_float_digits);
    double Ulimit_f= double(int (Ulimit*pow(10,m_float_digits)))/pow(10,m_float_digits);

    TString tag_text="Limits ";

    TString limit_str;
    limit_str+=Llimit_f;
    limit_str.ReplaceAll(" ","");
    tag_text+=limit_str;

    tag_text+=" - ";

    limit_str="";
    limit_str+=Ulimit_f;
    limit_str.ReplaceAll(" ","");
    tag_text+=limit_str;

    tag_text+=" at ";

    limit_str="";
    limit_str+=CL*100;
    limit_str.ReplaceAll(" ","");
    tag_text+=limit_str;

    tag_text+="% CL";

    m_cl_lines_tags[m_cl_lines_tags_num]=new TText(left_margin+tag_offset,
                                                   deltaNLL+tag_offset,
                                                   tag_text.Data());

    m_cl_lines_tags[m_cl_lines_tags_num]->SetTextSize(TEXT_SIZE);
    m_cl_lines_tags[m_cl_lines_tags_num]->SetTextColor(m_cl_lines_tags_num+3);

    ++m_cl_lines_tags_num;

    }

/*----------------------------------------------------------------------------*/

int PLScanPlot::addFCgraph(TGraphErrors* FC_graph, double CL){

    //    std::cout << "WARNING!!" << m_fc_graphs_num << std::endl;

    if (m_fc_graphs_num>=MAX_CL_LINES){
        std::cout << "INFO: Impossible to add another graph.Limit reached.\n";
        return -1;
        }

    TString graph_name="FC_graph_n";
    graph_name+=m_fc_graphs_num;
    graph_name.ReplaceAll(" ","");



    m_fc_graphs[m_fc_graphs_num]=
               dynamic_cast<TGraphErrors*> (FC_graph->Clone(graph_name.Data()));
    m_fc_graphs_num++;

    double left_margin=m_scan_graph->GetXaxis()->GetXmin();
    double right_margin=m_scan_graph->GetXaxis()->GetXmax();

    double deltaNLL= m_getDeltaNLLfromCL(CL);


    if (m_cl_lines_num>=MAX_CL_LINES){
        std::cout << "INFO: Impossible to add another line.Limit reached.\n";
        return -1;
        }

    m_cl_lines[m_cl_lines_num]=new TLine(left_margin, deltaNLL,
                                         right_margin, deltaNLL);

    m_cl_lines[m_cl_lines_num]->SetLineStyle(3);

    m_cl_lines_num++;

    }


/*----------------------------------------------------------------------------*/


double PLScanPlot::m_getDeltaNLLfromCL(double CL){
    double sqrtDeltaNll=TMath::ErfInverse (CL);
    return sqrtDeltaNll*sqrtDeltaNll;
    }

/*----------------------------------------------------------------------------*/

void PLScanPlot::m_build_minimum_tag(){
    // the minimum tag
    
    double fancy_min= double(int(m_val_at_min*pow(10,m_float_digits))
                        /pow(10,m_float_digits));

    TString tag_text = m_scanned_var;
    tag_text += " at minimum: ";
    TString fancy_min_str;
    fancy_min_str+=fancy_min;
    fancy_min_str.ReplaceAll(" ","");
    tag_text += fancy_min_str;

    m_minimum_tag = new TText(m_val_at_min,-1,tag_text.Data());
    m_minimum_tag->SetTextFont(12);
    }

/*----------------------------------------------------------------------------*/

void PLScanPlot::draw(const char* options){
    //gROOT->SetStyle("Plain");
    setCanvas(new TCanvas(GetName(),GetTitle()));
    getCanvas()->cd();
    getCanvas()->Draw(options);

    // The graphs
    m_scan_graph->Draw("APC");
    m_min_point_graph->Draw("P");

    // the 0 line
    m_zero_line->Draw("Same");

    // the min val tag 
    m_build_minimum_tag();
    m_minimum_tag->Draw("same");

    // the cl lines
    for (int i=0;i<m_cl_lines_num;++i){
        m_cl_lines[i]->Draw("Same");
        }

    // the cl tags
    for (int i=0;i<m_cl_lines_tags_num;++i){
        m_cl_lines_tags[i]->Draw("Same");
        }

    // the FC graphs
    for (int i=0;i<m_fc_graphs_num;++i){
        m_fc_graphs[i]->Draw("P");
        }


    }

/*----------------------------------------------------------------------------*/

void PLScanPlot::dumpToFile (const char* RootFileName, const char* options){
    TFile ofile(RootFileName,options);
    ofile.cd();
    m_scan_graph->Write();
    m_min_point_graph->Write();
    m_minimum_tag->Write("minvalTag");

    TString linename;
    for (int i=0;i<m_cl_lines_num;++i){
        linename="CLline_";
        linename+=i;
        m_cl_lines[i]->Write(linename.Data());
        }

    TString linetagname;
    for (int i=0;i<m_cl_lines_tags_num;++i){
        linetagname="CLlineTag_";
        linetagname+=i;
        m_cl_lines_tags[i]->Write(linetagname.Data());
        }

    ofile.Close();
    }

/*----------------------------------------------------------------------------*/

PLScanPlot::~PLScanPlot(){
    delete m_scan_graph;
    delete m_min_point_graph;
    delete m_zero_line;

    if (m_minimum_tag!=NULL)
        delete m_minimum_tag;

    for (int i=0;i<m_fc_graphs_num;++i)
        delete m_fc_graphs[i];

    for (int i=0;i<m_cl_lines_num;++i)
        delete m_cl_lines[i];

    for (int i=0;i<m_cl_lines_tags_num;++i)
        delete m_cl_lines_tags[i];

    }

/*----------------------------------------------------------------------------*/

void PLScanPlot::print(const char* options){
    std::cout << "\nPLScanPlot object " << GetName() << ":\n";
    }

/*----------------------------------------------------------------------------*/


/// To build the cint dictionaries
//ClassImp(PLScanPlot)

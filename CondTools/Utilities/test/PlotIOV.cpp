template <class T> void NoPlotIOVCache(CondCachedIter<T>* Iterator,unsigned int IDDet,const std::string & KindOfData,const std::string & WhatToPlot,const std::string & General){

    gROOT->SetStyle("Plain");

    std::vector<double> Run;
    std::vector<double> RunStart;
    std::vector<double> RunStop;
    std::vector<double> ValueY;
    
    const T* reference = 0;

    int counter = 0;
   
     
    while(reference = Iterator->next()) {
        unsigned int *SizeOfVector = new unsigned int;
         
        TString CommandToROOTSize = Form("const %s *pointer; pointer = (%s *) 0x%x; unsigned int *SizeIN = (unsigned int *) 0x%x; *SizeIN=((pointer->%s).size());",General.c_str(),General.c_str(),reference,SizeOfVector,KindOfData.c_str());
        
        gROOT->ProcessLine(CommandToROOTSize);
                
        if(*SizeOfVector > 0) {
            Run.push_back(Iterator->getTime());
            RunStart.push_back(Iterator->getStartTime());
            RunStop.push_back(Iterator->getStopTime());
        
 //---- Pointer access: all the following instructions are "only" to fill the vector ----
                    
            TString CommandToROOT = Form("const %s *pointer; pointer = (%s *) 0x%x; std::vector<double> *ValueYIN = (std::vector<double> *) 0x%x; unsigned int *IDDetIN = (unsigned int) 0x%x; ValueYIN.push_back(((pointer->%s)[*IDDetIN]).%s);",General.c_str(),General.c_str(),reference,&ValueY,&IDDet,KindOfData.c_str(),WhatToPlot.c_str());
        
            gROOT->ProcessLine(CommandToROOT);
            counter++;
        }
        delete SizeOfVector;
       

    }
        
    Run.clear();
    RunStart.clear();
    RunStop.clear();
    ValueY.clear();

}






template <class T> void NoPlotIOV(CondIter<T>* Iterator,unsigned int IDDet,const std::string & KindOfData,const std::string & WhatToPlot,const std::string & General){

    gROOT->SetStyle("Plain");

    std::vector<double> Run;
    std::vector<double> RunStart;
    std::vector<double> RunStop;
    std::vector<double> ValueY;
    
    const T* reference = 0;

    int counter = 0;
    
    while(reference = Iterator->next()) { 

        unsigned int *SizeOfVector = new unsigned int;
                
        TString CommandToROOTSize = Form("const %s *pointer; pointer = (%s *) 0x%x; unsigned int *SizeIN = (unsigned int *) 0x%x; *SizeIN=((pointer->%s).size());",General.c_str(),General.c_str(),reference,SizeOfVector,KindOfData.c_str());
        
        gROOT->ProcessLine(CommandToROOTSize);
                
        if(*SizeOfVector > 0) {
            Run.push_back(Iterator->getTime());
            RunStart.push_back(Iterator->getStartTime());
            RunStop.push_back(Iterator->getStopTime());
        
 //---- Pointer access: all the following instructions are "only" to fill the vector ----
                    
            TString CommandToROOT = Form("const %s *pointer; pointer = (%s *) 0x%x; std::vector<double> *ValueYIN = (std::vector<double> *) 0x%x; unsigned int *IDDetIN = (unsigned int) 0x%x; ValueYIN.push_back(((pointer->%s)[*IDDetIN]).%s);",General.c_str(),General.c_str(),reference,&ValueY,&IDDet,KindOfData.c_str(),WhatToPlot.c_str());
        
            gROOT->ProcessLine(CommandToROOT);
            counter++;
        }
        
        delete SizeOfVector;

    }
    
    
    Run.clear();
    RunStart.clear();
    RunStop.clear();
    ValueY.clear();

    
}









template <class T> void PlotIOV(CondIter<T>* Iterator,unsigned int IDDet,const std::string & KindOfData,const std::string & WhatToPlot,const std::string & General){

    gROOT->SetStyle("Plain");

    std::vector<double> Run;
    std::vector<double> RunStart;
    std::vector<double> RunStop;
    std::vector<double> ValueY;
    
    const T* reference;

    int counter = 0;

    while(reference = Iterator->next()) { 

        
        unsigned int SizeOfVector = 0;
        
        TString CommandToROOTSize = Form("const %s *pointer; pointer = (%s *) 0x%x; unsigned int *SizeIN = (unsigned int *) 0x%x; *SizeIN=((pointer->%s).size());",General.c_str(),General.c_str(),reference,&SizeOfVector,KindOfData.c_str());

        gROOT->ProcessLine(CommandToROOTSize);
        
        std::cout << "Executed " << SizeOfVector << " times" << std::endl;


        if(SizeOfVector > 0) {
            Run.push_back(Iterator->getTime());
            RunStart.push_back(Iterator->getStartTime());
            RunStop.push_back(Iterator->getStopTime());

        //---- Pointer access: all the following instructions are "only" to fill the vector ----
            
            TString CommandToROOT = Form("const %s *pointer; pointer = (%s *) 0x%x; std::vector<double> *ValueYIN = (std::vector<double> *) 0x%x; unsigned int *IDDetIN = (unsigned int) 0x%x; ValueYIN.push_back(((pointer->%s)[*IDDetIN]).%s);",General.c_str(),General.c_str(),reference,&ValueY,&IDDet,KindOfData.c_str(),WhatToPlot.c_str());

            gROOT->ProcessLine(CommandToROOT);
            counter++;
        }

    }


    std::cout << "\n\nGraph with Bars SINCE and TILL time\n\n";


    double *Time = new double[2*counter];
    double *Value = new double[2*counter];

    for(unsigned int i=0; i<counter; i++) {
        Time[2*i] = (double) RunStart.at(i);
        Time[2*i+1] = (double) RunStop.at(i) +1;// "+1" since each run lasts 1 unit
        Value[2*i] = ValueY.at(i);
        Value[2*i+1] = ValueY.at(i);
    }


    TCanvas *ccBar;
    TGraph *GraphBar;
    ccBar = new TCanvas ("Bar","Bar",10,10,700,400);

    GraphBar = new TGraph(2*counter,Time,Value);
    GraphBar->SetTitle("Graph X vs Run Bar");
    GraphBar->SetMarkerColor(4);
    GraphBar->SetMarkerSize(.3);
    GraphBar->SetMarkerStyle(20);
    GraphBar->SetLineColor(3);
    GraphBar->SetFillColor(3);
    GraphBar->Draw("APL");
    GraphBar->GetXaxis()->SetTitle("# Run");
    GraphBar->GetYaxis()->SetTitleOffset(1.2);
    GraphBar->GetYaxis()->SetTitle("X");
    GraphBar->GetXaxis()->SetTitleSize(0.04);
    GraphBar->GetYaxis()->SetTitleSize(0.04);
    ccBar->Update();

    Run.clear();
    RunStart.clear();
    RunStop.clear();
    ValueY.clear();
    
    
    //----
    delete Time;
    delete Value;

}






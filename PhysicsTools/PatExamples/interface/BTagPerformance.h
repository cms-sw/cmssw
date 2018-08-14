#ifndef BTagPerformance_h
#define BTagPerformance_h
/** \class BTagPerformance
 *
 * Analyze ROOT files produced by analyzer and create plots
 *
 * \author Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 *
 *
 */

#include <map>
#include <assert.h>
#include "TGraph.h"
#include "TArrayD.h"

class BTagPerformance {

  public:
        BTagPerformance() {};
        void Set(std::string name) {
                fname = name;
                fNcuts = 40;
                b_all = c_all = udsg_all = 0;
                for( int i=0; i<fNcuts; ++i) {
                  b_tagged[i] = 0;
                  c_tagged[i] = 0;
                  udsg_tagged[i] = 0;
                }
                
        };
        void SetMinDiscriminator( double value) { fMinDisc = value; };
        void SetMaxDiscriminator( double value) { fMaxDisc = value; };
        void SetNcuts( int value ) { 
          fNcuts = value;
          for( int i=0; i<fNcuts; ++i) {
            b_tagged[i] = 0;
            c_tagged[i] = 0;
            udsg_tagged[i] = 0;
          }
        };
        void Add( double discriminator, int flavor ) {

                fdisc = discriminator;
                fflavor = flavor;
                double binwidth = (fMaxDisc-fMinDisc)/((double) fNcuts);
                if (fflavor==5) b_all++;
                if (fflavor==4) c_all++;
                if ((fflavor>0 && fflavor<4)||(fflavor==21)) udsg_all++;
                
                for( int i=0; i<fNcuts; ++i ) {
                        double cut = i*binwidth + fMinDisc;
                        if (cut > fMaxDisc) cut = fMaxDisc;

                        disc_map[i] = cut;
                        if ( fdisc > cut ) {
                                if ( fflavor==5 ) b_tagged[i]++;
                                if ( fflavor==4 ) c_tagged[i]++;
                                if ((fflavor>0 && fflavor<4)||(fflavor==21)) udsg_tagged[i]++;
                        }
                }
        };

        void Eval() {
                double small = 1.e-5;
                int ip = 0;
                for(std::map<int,double>::const_iterator im=b_tagged.begin(); im!=b_tagged.end(); ++im) {

                        double eff = (im->second)/((double)b_all);
                        double err = sqrt(eff*(1-eff)/b_all);
                        if ( eff==0 || eff< small ) { eff = small; err=0; }
                        b_eff[ip] = eff;
                        b_effErr[ip] = err;
                        //g1["b"+fname].SetPoint(ip,eff,eff);
                        //h1["b"+fname]->Fill(eff);
                        ip++;
                }
                ip=0;
                for(std::map<int,double>::const_iterator im=c_tagged.begin(); im!=c_tagged.end(); ++im) {
                  
                  double eff = (im->second)/((double)c_all);
                  double err = sqrt(eff*(1-eff)/c_all);
                  if ( eff==0 || eff< small ) { eff = small; err=0; }
                  c_eff[ip] = eff;
                  c_effErr[ip] = err;
                  //g1["c"+fname].SetPoint(ip,b_tagged[ip],eff);
                        //h1["c"+fname]->Fill(eff);
                        ip++;
                }
                ip=0;
                for(std::map<int,double>::const_iterator im=udsg_tagged.begin(); im!=udsg_tagged.end(); 
++im) {

                        double eff = (im->second)/((double)udsg_all);
                        double err = sqrt(eff*(1-eff)/udsg_all);
                        if ( eff==0 || eff< small ) { eff = small; err=0; }
                        udsg_eff[ip] = eff;
                        udsg_effErr[ip] = err;
                        //g1["udsg"+fname].SetPoint(ip,b_tagged[ip],eff);
                        //h1["udsg"+fname]->Fill(eff);
                        ip++;
                }
                                
        };
        std::map< int,double> GetMap(TString option="b") {
          if (option=="b") return b_eff;
          if (option=="c") return c_eff;
          if (option=="udsg") return udsg_eff;
          if (option=="bErr") return b_effErr;
          if (option=="cErr") return c_effErr;
          if (option=="udsgErr") return udsg_effErr;
          if (option=="discriminator") return disc_map;
          else { assert(false); return disc_map;}
        };

        TArrayD GetArray(TString option="b") {
          std::map<int,double> amap = GetMap(option);
          TArrayD tarray(fNcuts);
          for(std::map<int,double>::const_iterator im=amap.begin(); im!=amap.end(); ++im) {
                  //std::cout << "i= " << im->first << " value= " << im->second << std::endl;
            tarray[im->first] = im->second;
          }
          return tarray;
          
        };

        int GetN() { return fNcuts; };
//ed++
        double Getb_all() {return b_all; }; 
        double Getc_all() {return c_all; }; 
        double Getudsg_all() {return udsg_all; }; 
//ed--
  private:
        Double_t *farray;
        double fdisc;
        int    fflavor;
        double fMinDisc;
        double fMaxDisc;
        int    fNcuts;
        std::string fname;
        //std::map< std::string, TH1* > h1;
        //std::map< std::string, TGraph > g1;
        std::map< int, double > b_eff;
        std::map< int, double > c_eff;
        std::map< int, double > udsg_eff;
        std::map< int, double > b_effErr;
        std::map< int, double > c_effErr;
        std::map< int, double > udsg_effErr;
        std::map< int, double > disc_map;
        
        std::map< int, double > b_tagged;
        std::map< int, double > c_tagged;
        std::map< int, double > udsg_tagged;
        double b_all;
        double c_all;
        double udsg_all;
        
        
        
};

#endif


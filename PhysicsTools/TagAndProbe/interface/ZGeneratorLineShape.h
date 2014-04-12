#ifndef Z_GENERATOR_LINE_SHAPE
#define Z_GENERATOR_LINE_SHAPE

#include "Riostream.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "TH1F.h"
#include "TFile.h"

class ZGeneratorLineShape : public RooAbsPdf {

public:
  ZGeneratorLineShape() {} ; 
  ZGeneratorLineShape(const char *name, const char *title,
		      RooAbsReal& _m,
                      //char* genfile = "/uscmst1b_scratch/lpc1/old_scratch/lpctrig/jwerner/ZeePASResConvFitter/ZeeGenLevel.root"
                      const char* genfile = "ZeeGenLevel.root", const char* histoName= "Mass"
		      );

  ZGeneratorLineShape(const ZGeneratorLineShape& other, const char* name);
  inline virtual TObject* clone(const char* newname) const { return new ZGeneratorLineShape(*this,newname);}
  inline ~ZGeneratorLineShape(){};
  ClassDef(ZGeneratorLineShape,1)
    Double_t evaluate() const;  
 protected:
  RooRealProxy m ;
  RooDataHist* dataHist;
};

#endif

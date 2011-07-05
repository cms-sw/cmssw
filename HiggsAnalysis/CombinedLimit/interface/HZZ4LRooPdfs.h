#ifndef HZZ4LROOPDFS
#define HZZ4LROOPDFS

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"

#include <iostream>
#include <fstream>
#include <string>

/*
namespace RooFit{
	
	void readFile();
	
	const Double_t FracEventsNoBrem_4mu = 0.703618;
	const Double_t FracEventsNoBrem_4e = 0.583196;
	const Double_t FracEventsNoBrem_2e2mu = 0.641297;
	
	Double_t Lgg_7(Double_t mHstar);
	Double_t HiggsWidth(Int_t ID,Double_t mHrequested);
	Double_t pdf1(double mHstar,double mHreq);
	Double_t rho(double r, TString proc);
	Double_t Sigma(double mHreq, TString proc);
	
	Double_t N(Double_t mH, TString proc);
	Double_t sigma_CB(Double_t mH, TString proc);
	Double_t mean(Double_t mH, TString proc);
	
	Double_t scratchMass;
	Double_t BR[26][217];
	Double_t CS[6][197];
	
}
*/


class RooqqZZPdf : public RooAbsPdf {
public:
	RooqqZZPdf() {} ;
	RooqqZZPdf(const char *name, const char *title,
			   RooAbsReal& _m4l,
			   RooAbsReal& _a1,
			   RooAbsReal& _a2,
			   RooAbsReal& _a3,
			   RooAbsReal& _b1,
			   RooAbsReal& _b2,
			   RooAbsReal& _b3,
			   RooAbsReal& _frac);
	RooqqZZPdf(const RooqqZZPdf& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooqqZZPdf(*this,newname); }
	inline virtual ~RooqqZZPdf() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy a1 ;
	RooRealProxy a2 ;
	RooRealProxy a3 ;
	RooRealProxy b1 ;
	RooRealProxy b2 ;
	RooRealProxy b3 ;
	RooRealProxy frac ;
	
	Double_t evaluate() const ;
	
private:
	
	ClassDef(RooqqZZPdf,1) // Your description goes here...                                                                                                   
};



class RooggZZPdf : public RooAbsPdf {
public:
	RooggZZPdf() {} ;
	RooggZZPdf(const char *name, const char *title,
			   RooAbsReal& _m4l,
			   RooAbsReal& _a1,
			   RooAbsReal& _a2,
			   RooAbsReal& _a3,
			   RooAbsReal& _b1,
			   RooAbsReal& _b2,
			   RooAbsReal& _b3,
			   RooAbsReal& _frac);
	RooggZZPdf(const RooggZZPdf& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooggZZPdf(*this,newname); }
	inline virtual ~RooggZZPdf() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy a1 ;
	RooRealProxy a2 ;
	RooRealProxy a3 ;
	RooRealProxy b1 ;
	RooRealProxy b2 ;
	RooRealProxy b3 ;
	RooRealProxy frac ;
	
	Double_t evaluate() const ;
	
private:
	
	ClassDef(RooggZZPdf,1) // Your description goes here...                                                                                                   
};



class RooFourMuMassShapePdf2 : public RooAbsPdf {
public:
	RooFourMuMassShapePdf2() {} ;
	RooFourMuMassShapePdf2(const char *name, const char *title,
						   RooAbsReal& _m4l,
						   RooAbsReal& _mH);
	RooFourMuMassShapePdf2(const RooFourMuMassShapePdf2& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooFourMuMassShapePdf2(*this,newname); }
	inline virtual ~RooFourMuMassShapePdf2() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy mH ;
	
	Double_t evaluate() const ;
	//void readFile() const ;
	
private:
	
	std::string FileLoc;
	
	ClassDef(RooFourMuMassShapePdf2,1) // Your description goes here...                                                                                       
};


class RooFourEMassShapePdf2 : public RooAbsPdf {
public:
	RooFourEMassShapePdf2() {} ;
	RooFourEMassShapePdf2(const char *name, const char *title,
						  RooAbsReal& _m4l,
						  RooAbsReal& _mH);
	RooFourEMassShapePdf2(const RooFourEMassShapePdf2& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooFourEMassShapePdf2(*this,newname); }
	inline virtual ~RooFourEMassShapePdf2() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy mH ;
	
	Double_t evaluate() const ;
	//void readFile() const ;
	
private:
	
	std::string FileLoc;
	
	ClassDef(RooFourEMassShapePdf2,1) // Your description goes here...                                                                                        
};



class RooTwoETwoMuMassShapePdf2 : public RooAbsPdf {
public:
	RooTwoETwoMuMassShapePdf2() {} ;
	RooTwoETwoMuMassShapePdf2(const char *name, const char *title,
							  RooAbsReal& _m4l,
							  RooAbsReal& _mH);
	RooTwoETwoMuMassShapePdf2(const RooTwoETwoMuMassShapePdf2& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooTwoETwoMuMassShapePdf2(*this,newname); }
	inline virtual ~RooTwoETwoMuMassShapePdf2() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy mH ;
	
	Double_t evaluate() const ;
	//void readFile() const ;
	
private:
	
	std::string FileLoc;
	
	
	ClassDef(RooTwoETwoMuMassShapePdf2,1) // Your description goes here...                                                                                    
};


class RooFourMuMassRes : public RooAbsPdf {
public:
	RooFourMuMassRes() {} ;
	RooFourMuMassRes(const char *name, const char *title,
					 RooAbsReal& _m4l,
					 RooAbsReal& _mH);
	RooFourMuMassRes(const RooFourMuMassRes& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooFourMuMassRes(*this,newname); }
	inline virtual ~RooFourMuMassRes() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy mH  ;
	
	Double_t evaluate() const ;
	
private:
	
	ClassDef(RooFourMuMassRes,1) // Your description goes here...                                                                                             
};

class RooFourEMassRes : public RooAbsPdf {
public:
	RooFourEMassRes() {} ;
	RooFourEMassRes(const char *name, const char *title,
					RooAbsReal& _m4l,
					RooAbsReal& _mH);
	RooFourEMassRes(const RooFourEMassRes& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooFourEMassRes(*this,newname); }
	inline virtual ~RooFourEMassRes() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy mH  ;
	
	Double_t evaluate() const ;
	
private:
	
	ClassDef(RooFourEMassRes,1) // Your description goes here...                                                                                              
};


class RooTwoETwoMuMassRes : public RooAbsPdf {
public:
	RooTwoETwoMuMassRes() {} ;
	RooTwoETwoMuMassRes(const char *name, const char *title,
						RooAbsReal& _m4l,
						RooAbsReal& _mH);
	RooTwoETwoMuMassRes(const RooTwoETwoMuMassRes& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooTwoETwoMuMassRes(*this,newname); }
	inline virtual ~RooTwoETwoMuMassRes() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy mH  ;
	
	
	Double_t evaluate() const ;
	
private:
	
	ClassDef(RooTwoETwoMuMassRes,1) // Your description goes here...                                                                                          
};

class RooRelBW1 : public RooAbsPdf {
public:
	RooRelBW1() {} ;
	RooRelBW1(const char *name, const char *title,
			  RooAbsReal& _m,
			  RooAbsReal& _mean,
			  RooAbsReal& _gamma);
	RooRelBW1(const RooRelBW1& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooRelBW1(*this,newname); }
	inline virtual ~RooRelBW1() { }
	
protected:
	
	RooRealProxy m ;
	RooRealProxy mean ;
	RooRealProxy gamma ;
	
	Double_t evaluate() const ;
	
private:
	
	ClassDef(RooRelBW1,1) // Your description goes here...                                                                                                    
};

//////////////////////////////////////////////

class RooRelBWUF : public RooAbsPdf {
public:
	RooRelBWUF() {} ;
	RooRelBWUF(const char *name, const char *title,
			  RooAbsReal& _m4l,
			  RooAbsReal& _mH);
	RooRelBWUF(const RooRelBWUF& other, const char* name=0) ;
	virtual TObject* clone(const char* newname) const { return new RooRelBWUF(*this,newname); }
	inline virtual ~RooRelBWUF() { }
	
protected:
	
	RooRealProxy m4l ;
	RooRealProxy mH ;
	
	Double_t evaluate() const ;
	
private:
	
	std::string FileLoc;
	
	ClassDef(RooRelBWUF,1) // Your description goes here...                                                                                                    
};





#endif

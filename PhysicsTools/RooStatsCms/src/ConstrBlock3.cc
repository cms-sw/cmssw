// @(#)root/hist:$Id: ConstrBlock3.cc,v 1.1 2009/01/06 12:22:43 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"

#include "RooFormulaVar.h"

#include "PhysicsTools/RooStatsCms/interface/ConstrBlock3.h"
#include "PhysicsTools/RooStatsCms/interface/ConstrBlock2.h"



/*----------------------------------------------------------------------------*/

/**
Constructor of the class
**/

ConstrBlock3::ConstrBlock3 (const char* the_name,
                            const char* title,
                            double corr12,
                            double corr13,
                            double corr23,
                            Constraint* constr1,
                            Constraint* constr2,
                            Constraint* constr3):
    TNamed(the_name,title){

    setVerbosity(true);

    TString name(GetName());

    m_constr_list =  new RooArgList("Constr list");
    m_constr_list->add(*constr1);
    m_constr_list->add(*constr2);
    m_constr_list->add(*constr3);


    // The correlation terms
    TString corr_name=name;
    corr_name+="_corr12";
    m_corr[0]=new RooRealVar(corr_name.Data(),
                             corr_name.Data(),
                             corr12);

    corr_name=GetName();
    corr_name+="_corr13";
    m_corr[1]=new RooRealVar(corr_name.Data(),
                             corr_name.Data(),
                             corr13);

    corr_name=GetName();
    corr_name+="_corr23";
    m_corr[2]=new RooRealVar(corr_name.Data(),
                             corr_name.Data(),
                             corr23);



    // The list to store the parameters involved
    m_parameters = new RooArgList("Parameters");

    // With this addition we have params, means and sigmas in the list
    // We are sure that for gaussians the second is the mean and the third the 
    // sigma
    m_parameters->add(constr1->getNLLterms());
    m_parameters->add(constr2->getNLLterms());
    m_parameters->add(constr3->getNLLterms());

    RooRealVar* mean1=static_cast<RooRealVar*>(&(constr1->getNLLterms())[1]);
    RooRealVar* mean2=static_cast<RooRealVar*>(&(constr2->getNLLterms())[1]);
    RooRealVar* mean3=static_cast<RooRealVar*>(&(constr3->getNLLterms())[1]);

    RooRealVar* sigma1=static_cast<RooRealVar*>(&(constr1->getNLLterms())[2]);
    RooRealVar* sigma2=static_cast<RooRealVar*>(&(constr2->getNLLterms())[2]);
    RooRealVar* sigma3=static_cast<RooRealVar*>(&(constr3->getNLLterms())[2]);


    TString xm1_n="xm1_"+name;
    RooFormulaVar* xm1=new RooFormulaVar (xm1_n.Data(),
                                          "@0-@1",
                                          RooArgList(*constr1,*mean1));

    TString xm2_n="xm2_"+name;
    RooFormulaVar* xm2=new RooFormulaVar (xm2_n.Data(),
                                          "@0-@1",
                                          RooArgList(*constr2,*mean2));


    TString xm3_n="xm3_"+name;
    RooFormulaVar* xm3=new RooFormulaVar (xm3_n.Data(),
                                          "@0-@1",
                                          RooArgList(*constr3,*mean3));

    m_parameters->add(RooArgList(*xm1,*xm2,*xm3));

    // The covariance matrix
    TString V11_n="V11_"+name;
    RooFormulaVar* V11=new RooFormulaVar(V11_n.Data(),"@0**2",RooArgList(*sigma1));
    TString V22_n="V22_"+name;
    RooFormulaVar* V22=new RooFormulaVar(V22_n.Data(),"@0**2",RooArgList(*sigma2));
    TString V33_n="V33_"+name;
    RooFormulaVar* V33=new RooFormulaVar(V33_n.Data(),"@0**2",RooArgList(*sigma3));
    TString V12_n="V12_"+name;
    RooFormulaVar* V12=new RooFormulaVar(V12_n.Data(),"@0*@1*@2",RooArgList(*sigma1,*sigma2,*m_corr[0]));
    TString V13_n="V13_"+name;
    RooFormulaVar* V13=new RooFormulaVar(V13_n.Data(),"@0*@1*@2",RooArgList(*sigma1,*sigma3,*m_corr[1]));
    TString V23_n="V23_"+name;
    RooFormulaVar* V23=new RooFormulaVar(V23_n,"@0*@1*@2",RooArgList(*sigma2,*sigma3,*m_corr[2]));

    RooArgList vlist(*V11,*V22,*V33,*V12,*V13,*V23);
    m_parameters->add(vlist);

    std::cout << "The Sigmas:\n";
    sigma1->Print();
    sigma2->Print();
    sigma3->Print();

    RooFormulaVar* vs[6]={V11,V22,V33,V12,V13,V23};
    for (int i=0;i<6;++i)
        std::cout << "  - V["<<i<<"]" << vs[i]->getVal() << std::endl;

    // Its determinant inverse
    TString invDetV_n="invDetV_"+name;
    TString formula="1/("+V11_n+"*("+V22_n+"*"+V33_n+"-"+V23_n+"*"+V23_n+")-"+
                          V12_n+"*("+V12_n+"*"+V33_n+"-"+V23_n+"*"+V13_n+")+"+
                          V13_n+"*("+V12_n+"*"+V23_n+"-"+V22_n+"*"+V13_n+"))";

//     RooFormulaVar* invDetV=new RooFormulaVar(invDetV_n.Data(),
//         "1/(V11*(V22*V33-V23*V23)-V12*(V12*V33-V23*V13)+V13*(V12*V23-V22*V13))",
//          vlist);
    RooFormulaVar* invDetV=new RooFormulaVar(invDetV_n.Data(),
                                             formula.Data(),
                                             vlist);

    m_parameters->add(*invDetV);
    vlist.add(*invDetV);
/*
    // inverse of the covariance matrix U = V^-1
    TString U11_n="U11_"+name;
    formula=invDetV_n+"*("+V22_n+"*"+V33_n+"-"+V23_n+"*"+V23_n+")";
    RooFormulaVar* U11=new RooFormulaVar(U11_n.Data(),formula.Data(),vlist);
    TString U12_n="U12_"+name;
    formula=invDetV_n+"*("+V13_n+"*"+V23_n+"-"+V33_n+"*"+V12_n+")";
    RooFormulaVar* U12=new RooFormulaVar(U12_n.Data(),formula.Data(),vlist);
    TString U13_n="U13_"+name;
    formula=invDetV_n+"*("+V12_n+"*"+V23_n+"-"+V22_n+"*"+V13_n+")";
    RooFormulaVar* U13=new RooFormulaVar(U13_n.Data(),formula.Data(),vlist);
    TString U21_n="U21_"+name;
    formula=invDetV_n+"*("+V23_n+"*"+V13_n+"-"+V33_n+"*"+V12_n+")";
    RooFormulaVar* U21=new RooFormulaVar(U21_n.Data(),formula.Data(),vlist);
    TString U22_n="U22_"+name;
    formula=invDetV_n+"*("+V11_n+"*"+V33_n+"-"+V13_n+"*"+V13_n+")";
    RooFormulaVar* U22=new RooFormulaVar(U22_n.Data(),formula.Data(),vlist);
    TString U23_n="U23_"+name;
    formula=invDetV_n+"*("+V13_n+"*"+V12_n+"-"+V23_n+"*"+V11_n+")";
    RooFormulaVar* U23=new RooFormulaVar(U23_n.Data(),formula.Data(),vlist);
    TString U31_n="U31_"+name;
    formula=invDetV_n+"*("+V12_n+"*"+V23_n+"-"+V13_n+"*"+V22_n+")";
    RooFormulaVar* U31=new RooFormulaVar(U31_n.Data(),formula.Data(),vlist);
    TString U32_n="U32_"+name;
    formula=invDetV_n+"*("+V12_n+"*"+V13_n+"-"+V23_n+"*"+V11_n+")";
    RooFormulaVar* U32=new RooFormulaVar(U32_n.Data(),formula.Data(),vlist);
    TString U33_n="U33_"+name;
    formula=invDetV_n+"*("+V11_n+"*"+V22_n+"-"+V12_n+"*"+V12_n+")";
    RooFormulaVar* U33=new RooFormulaVar(U33_n.Data(),formula.Data(),vlist);
*/
    TString invDetV_val("");
    invDetV_val+=invDetV->getVal();
    invDetV_val.ReplaceAll(" ","");

    // inverse of the covariance matrix U = V^-1
    TString U11_n="U11_"+name;
    formula=invDetV_val+"*("+V22_n+"*"+V33_n+"-"+V23_n+"*"+V23_n+")";
    RooFormulaVar* U11=new RooFormulaVar(U11_n.Data(),formula.Data(),vlist);
    TString U12_n="U12_"+name;
    formula=invDetV_val+"*("+V13_n+"*"+V23_n+"-"+V33_n+"*"+V12_n+")";
    RooFormulaVar* U12=new RooFormulaVar(U12_n.Data(),formula.Data(),vlist);
    TString U13_n="U13_"+name;
    formula=invDetV_val+"*("+V12_n+"*"+V23_n+"-"+V22_n+"*"+V13_n+")";
    RooFormulaVar* U13=new RooFormulaVar(U13_n.Data(),formula.Data(),vlist);
    TString U21_n="U21_"+name;
    formula=invDetV_val+"*("+V23_n+"*"+V13_n+"-"+V33_n+"*"+V12_n+")";
    RooFormulaVar* U21=new RooFormulaVar(U21_n.Data(),formula.Data(),vlist);
    TString U22_n="U22_"+name;
    formula=invDetV_val+"*("+V11_n+"*"+V33_n+"-"+V13_n+"*"+V13_n+")";
    RooFormulaVar* U22=new RooFormulaVar(U22_n.Data(),formula.Data(),vlist);
    TString U23_n="U23_"+name;
    formula=invDetV_val+"*("+V13_n+"*"+V12_n+"-"+V23_n+"*"+V11_n+")";
    RooFormulaVar* U23=new RooFormulaVar(U23_n.Data(),formula.Data(),vlist);
    TString U31_n="U31_"+name;
    formula=invDetV_val+"*("+V12_n+"*"+V23_n+"-"+V13_n+"*"+V22_n+")";
    RooFormulaVar* U31=new RooFormulaVar(U31_n.Data(),formula.Data(),vlist);
    TString U32_n="U32_"+name;
    formula=invDetV_val+"*("+V12_n+"*"+V13_n+"-"+V23_n+"*"+V11_n+")";
    RooFormulaVar* U32=new RooFormulaVar(U32_n.Data(),formula.Data(),vlist);
    TString U33_n="U33_"+name;
    formula=invDetV_val+"*("+V11_n+"*"+V22_n+"-"+V12_n+"*"+V12_n+")";
    RooFormulaVar* U33=new RooFormulaVar(U33_n.Data(),formula.Data(),vlist);


    m_parameters->add(RooArgList(*U11,*U12,*U13,*U21,*U22,*U23,*U31,*U32,*U33));


//     m_NLL_string  = "+0.5*(xm1*(U11*xm1+U12*xm2+U13*xm3))";
//     m_NLL_string += "+0.5*(xm2*(U21*xm1+U22*xm2+U23*xm3))";
//     m_NLL_string += "+0.5*(xm3*(U31*xm1+U32*xm2+U33*xm3))";

//     m_NLL_string = "+0.5*("+xm1_n+ "*("+U11_n+"*"+xm1_n+ "+"+U12_n+"*"+xm2_n+ "+"+U13_n+"*"+xm3_n+"))";
//     m_NLL_string+= "+0.5*("+xm2_n+ "*("+U21_n+"*"+xm1_n+ "+"+U22_n+"*"+xm2_n+ "+"+U23_n+"*"+xm3_n+"))";
//     m_NLL_string+= "+0.5*("+xm3_n+ "*("+U31_n+"*"+xm1_n+ "+"+U32_n+"*"+xm2_n+ "+"+U33_n+"*"+xm3_n+"))";

    m_NLL_string = "+0.5*("+xm1_n+ "*("+U11_n+"*"+xm1_n+ "+"+U12_n+"*"+xm2_n+ "+"+U13_n+"*"+xm3_n+"))";
    m_NLL_string+= "+0.5*("+xm2_n+ "*("+U21_n+"*"+xm1_n+ "+"+U22_n+"*"+xm2_n+ "+"+U23_n+"*"+xm3_n+"))";
    m_NLL_string+= "+0.5*("+xm3_n+ "*("+U31_n+"*"+xm1_n+ "+"+U32_n+"*"+xm2_n+ "+"+U33_n+"*"+xm3_n+"))";
/*
    m_NLL_string  = "+0.5*(xm1*(U11*xm1+U12*xm2+U13*xm3))";
    m_NLL_string += "+0.5*(xm2*(U21*xm1+U22*xm2+U23*xm3))";
    m_NLL_string += "+0.5*(xm3*(U31*xm1+U32*xm2+U33*xm3))";
*/

    std::cout << "Inv Det " << invDetV->getVal() << endl;

    RooFormulaVar* us[9]={U11,U12,U13,U21,U22,U23,U31,U32,U33};
    for (int i=0;i<9;++i)
        std::cout << "  - " << us[i]->getVal() << std::endl;



/*CANNOT EVAL PROPERLY - TOO COMPLICATED


    // the list of the constraints involved
    m_constr_list =  new RooArgList("Constr list");

    m_constr_list->add(*constr1);
    m_constr_list->add(*constr2);
    m_constr_list->add(*constr3);

    // Calculate the NLL string terms

    // With this addition we have params, means and sigmas in the list
    m_parameters->add(constr1->getNLLterms());
    m_parameters->add(constr2->getNLLterms());
    m_parameters->add(constr3->getNLLterms());

    RooRealVar* corr_12 = new RooRealVar ("corr_12","",corr12);
    RooRealVar* corr_13 = new RooRealVar ("corr_13","",corr13);
    RooRealVar* corr_23 = new RooRealVar ("corr_23","",corr23);

    // now also the correlations..We are done!
    m_parameters->add(RooArgList(*corr_12,*corr_13,*corr_23));

    // Now build the coefficients as strings

    TString sigma1=constr1->GetName();
    sigma1+="_gaussian_constr_sigma";

    TString sigma2=constr2->GetName();
    sigma2+="_gaussian_constr_sigma";

    TString sigma3=constr3->GetName();
    sigma3+="_gaussian_constr_sigma";

    // covariance matrix V  ( Vij = Vji )

    TString V11=sigma1+"^2";
    TString V22=sigma2+"^2";
    TString V33=sigma3+"^2";
    TString V12=sigma1+"*"+sigma2+"*corr_12";
    TString V13=sigma1+"*"+sigma3+"*corr_13";
    TString V23=sigma2+"*"+sigma3+"*corr_23";

    TString invDetV="1/("+V11+"*("+V22+"*"+V33+"-"+V23+"*"+V13+")-"
                         +V12+"*("+V12+"*"+V33+"-"+V23+"*"+V13+")+"
                         +V13+"*("+V12+"*"+V23+"-"+V22+"*"+V13+"))";

    // inverse of the covariance matrix *U = V^-1
    TString *U11=invDetV+"*"+"("+V22+"*"+V33+"-"+V23+"*"+V23+")";
    TString *U12=invDetV+"*"+"("+V13+"*"+V23+"-"+V33+"*"+V12+")";
    TString *U13=invDetV+"*"+"("+V22+"*"+V33+"-"+V23+"*"+V23+")";

    TString *U21=invDetV+"*"+"("+V23+"*"+V13+"-"+V33+"*"+V12+")";
    TString *U22=invDetV+"*"+"("+V11+"*"+V33+"-"+V13+"*"+V13+")";
    TString *U23=invDetV+"*"+"("+V13+"*"+V12+"-"+V23+"*"+V11+")";

    TString *U31=invDetV+"*"+"("+V12+"*"+V23+"-"+V13+"*"+V22+")";
    TString *U32=invDetV+"*"+"("+V12+"*"+V13+"-"+V23+"*"+V11+")";
    TString *U33=invDetV+"*"+"("+V11+"*"+V22+"-"+V12+"*"+V12+")";


    
    Since the coefficients are constant and the likelihood term would be 
    far too big if all the term woul d be explicit, we take constant *Us
    

    RooFormulaVar* *U11f=new RooFormulaVar("*U11f",*U11.Data(),*m_parameters);
    RooFormulaVar* *U12f=new RooFormulaVar("*U12f",*U12.Data(),*m_parameters);
    RooFormulaVar* *U13f=new RooFormulaVar("*U13f",*U13.Data(),*m_parameters);

    RooFormulaVar* *U21f=new RooFormulaVar("*U21f",*U21.Data(),*m_parameters);
    RooFormulaVar* *U22f=new RooFormulaVar("*U22f",*U22.Data(),*m_parameters);
    RooFormulaVar* *U23f=new RooFormulaVar("*U23f",*U23.Data(),*m_parameters);

    RooFormulaVar* *U31f=new RooFormulaVar("*U31f",*U31.Data(),*m_parameters);
    RooFormulaVar* *U32f=new RooFormulaVar("*U32f",*U32.Data(),*m_parameters);
    RooFormulaVar* *U33f=new RooFormulaVar("*U33f",*U33.Data(),*m_parameters);

    m_parameters->add(RooArgList(**U11f,**U12f,**U13f,
                                 **U21f,**U22f,**U23f,
                                 **U31f,**U32f,**U33f));

//DEBUG
//     RooRealVar* U11v=new RooRealVar("U11v","",U11f->getVal());
//     RooRealVar* U12v=new RooRealVar("U12v","",U12f->getVal());
//     RooRealVar* U13v=new RooRealVar("U13v","",U13f->getVal());
// 
//     RooRealVar* U21v=new RooRealVar("U21v","",U21f->getVal());
//     RooRealVar* U22v=new RooRealVar("U22v","",U22f->getVal());
//     RooRealVar* U23v=new RooRealVar("U23v","",U23f->getVal());
// 
//     RooRealVar* U31v=new RooRealVar("U31v","",U31f->getVal());
//     RooRealVar* U32v=new RooRealVar("U32v","",U32f->getVal());
//     RooRealVar* U33v=new RooRealVar("U33v","",U33f->getVal());
// 
//     m_parameters->add(RooArgList(*U11v,*U12v,*U13v,
//                                  *U21v,*U22v,*U23v,
//                                  *U31v,*U32v,*U33v));
//DEBUG END



    TString x=constr1->GetName();
    TString x0=constr1->GetName();
    x0+="_gaussian_constr_mean";

    TString y=constr2->GetName();
    TString y0=constr2->GetName();
    y0+="_gaussian_constr_mean";

    TString z=constr3->GetName();
    TString z0=constr3->GetName();
    z0+="_gaussian_constr_mean";


    TString dx="("+x+"-"+x0+")";
    TString dy="("+y+"-"+y0+")";
    TString dz="("+z+"-"+z0+")";

    // add term: +1/2 (x-mu)^T V^-1 (x-mu)

   m_NLL_string=" +0.5*"+dx+"*(U11f*"+dx+"+U12f*"+dy+"+U13f*"+dz+")";
   m_NLL_string+="+0.5*"+dy+"*(U21f*"+dx+"+U22f*"+dy+"+U23f*"+dz+")";
   m_NLL_string+="+0.5*"+dz+"*(U31f*"+dx+"+U32f*"+dy+"+U33f*"+dz+")";

//    m_NLL_string=" +0.5*("+dx+"*(U11v*"+dx+"+U12v*"+dy+"+U13v*"+dz+")";
//    m_NLL_string+="+"+dy+"*(U21v*"+dx+"+U22v*"+dy+"+U23v*"+dz+")";
//    m_NLL_string+="+"+dz+"*(U31v*"+dx+"+U32v*"+dy+"+U33v*"+dz+"))";

//DEBUG
//     RooFormulaVar *dxf=new RooFormulaVar ("dxf","",dx,*m_parameters);
//     RooFormulaVar *dyf=new RooFormulaVar ("dyf","",dy,*m_parameters);
//     RooFormulaVar *dzf=new RooFormulaVar ("dzf","",dz,*m_parameters);
// 
//     m_parameters->add(RooArgList(*dxf,*dyf,*dzf));
// 
//     m_NLL_string=" +0.5*(dxf*(U11v*dxf+U12v*dyf+U13v*dzf)";
//     m_NLL_string+="+dyf*(U21v*dxf+U22v*dyf+U23v*dzf)";
//     m_NLL_string+="+dzf*(U31v*dxf+U32v*dyf+U33v*dzf))";

   m_NLL_string=" +0.5*("+dx+"*(*U11f*"+dx+"+*U12f*"+dy+"+*U13f*"+dz+"))";
   m_NLL_string+="+0.5*("+dy+"*(*U21f*"+dx+"+*U22f*"+dy+"+*U23f*"+dz+"))";
   m_NLL_string+="+0.5*("+dz+"*(*U31f*"+dx+"+*U32f*"+dy+"+*U33f*"+dz+"))";
CANNOT EVAL PROPERLY - TOO COMPLICATED*/ 



//DEBUG END

    // Now Build the same quantities for Background

    // Find the constraints related to the Background
    Constraint* c[3];
    int number_of_bkg;
    double corr_block2x2;


    m_getBkgConstraints(c,&number_of_bkg,&corr_block2x2);

    std::cout << "BKG COMPONENTS is " << number_of_bkg << std::endl;
    std::cout << "CORRELATION is : " << corr_block2x2<< std::endl;


    // 3 bkg
    if (number_of_bkg==3){
        m_Bkg_NLL_string=m_NLL_string;
        m_Bkg_parameters=new RooArgList(*m_parameters);
        }
    // 0 bkg
    else if (number_of_bkg==0){
        m_Bkg_NLL_string=" + 0";
        m_Bkg_parameters= new RooArgList();
        }


    // 1 bkg
    else if (number_of_bkg==1){
        for (int i=0;i<3;++i)
            if (c[i]!=NULL){
                m_Bkg_NLL_string=c[i]->getNLLstring();
                m_Bkg_parameters= new RooArgList(c[i]->getNLLterms());
                }
        }

    // 2 bkg
    else if (number_of_bkg==2){
        Constraint *c1,*c2;
        c1=c2=0;
        for (int i=0;i<3;++i)
            if (c[i]!=NULL){
                if (c1==0)
                    c1=c[i];
                else
                    c2=c[i];
                }
        // instance of a 2x2 constraint block
        ConstrBlock2* block2 = new ConstrBlock2("dummy","",corr_block2x2,c1,c2);

        m_Bkg_NLL_string=block2->getNLLstring();
        m_Bkg_parameters= new RooArgList(block2->getNLLterms());
        }
  }

/*----------------------------------------------------------------------------*/

void ConstrBlock3::print(const char* options){
    std::cout << "ConstrBlock " << GetName() << ":\n"
              << "\nParameters:\n";
    RooRealVar* par;
    for (int i=0;i<m_parameters->getSize();++i){
        par=(RooRealVar*) &((*m_parameters)[i]);
        std::cout << " - " << par->ClassName() << " object " << par->GetName()
                  << " " << par->getVal() << std::endl;
        }
    }

/*----------------------------------------------------------------------------*/

void ConstrBlock3::restore(){
    Constraint* constr;
    for (int i=0;i<m_constr_list->getSize();++i){
        constr=(Constraint*) &((*m_constr_list)[i]);
        constr->restore();
        }
    }

/*----------------------------------------------------------------------------*/

/**
We realy heavily on the orders of the input parameters:
Correlation-constr1-mean-sigma-constr2-mean-sigma
**/

void ConstrBlock3::fluctuate(){

    Constraint* constr1=(Constraint*)&(*m_constr_list)[0];
    Constraint* constr2=(Constraint*)&(*m_constr_list)[1];
    Constraint* constr3=(Constraint*)&(*m_constr_list)[2];


    double m[3][3];

    // Fill the matrix
    double sigma[3];

    sigma[0]=((RooRealVar*)&((constr1->getNLLterms())[2]))->getVal();
    sigma[1]=((RooRealVar*)&((constr2->getNLLterms())[2]))->getVal();;
    sigma[2]=((RooRealVar*)&((constr3->getNLLterms())[2]))->getVal();;

    // diagonal terms
    for (int i=0;i<3;++i)
        m[i][i]=sigma[i]*sigma[i];

    // non diagonal terms
    m[1][0]=m[0][1]=sigma[0]*sigma[1]*m_corr[0]->getVal();
    m[2][0]=m[0][2]=sigma[0]*sigma[2]*m_corr[1]->getVal();
    m[1][2]=m[2][1]=sigma[1]*sigma[2]*m_corr[2]->getVal();


    // Correlated generation
    // Create the as:
    float a11=sqrt(m[0][0]);

    float a21=m[1][0]/a11;
    float a22=sqrt(m[1][1]-a21*a21);

    float a31=m[0][2]/a11;
    float a32=(m[1][2]-a21*a31)/a22;
    float a33=sqrt(m[2][2]-a31*a31-a32*a32);

    if (is_verbose())
    std::cout << "[ConstrBlock3::fluctuate] "
              << "The a_ij:\n"
              << " - a11 = " << a11 << std::endl
              << " - a22 = " << a22 << std::endl
              << " - a21 = " << a21 << std::endl
              << " - a31 = " << a31 << std::endl
              << " - a32 = " << a32 << std::endl
              << " - a33 = " << a33 << std::endl;

    // Shoot 3 random numbers centered in the mean of the constraint
    float u[3];
    if (is_verbose())
        std::cout << "[ConstrBlock3::fluctuate] "
                  << " The us:\n";
    for(int i=0;i<3;i++){
        //u[i]=RANDOM_GENERATOR.Gaus(m_constraints_means_original[i],1);
        u[i]=gRandom->Gaus(0,1);
        if (is_verbose())
            std::cout << " - u[i]" << u[i]<< std::endl;
        }

    // and finally the 3 numbers!
    double y[3];
    y[0]=a11*u[0]+constr1->getOriginalValue();
    y[1]=a21*u[0]+a22*u[1]+constr2->getOriginalValue();
    y[2]=a31*u[0]+a32*u[1]+a33*u[2]+constr3->getOriginalValue();


    if (is_verbose())
        std::cout << "[ConstrBlock3::fluctuate] \n"
                  << "  - constr1_val : "
                  << constr1->getOriginalValue() << " --- " << y[0] << std::endl
                  << "  - constr2_val : "
                  << constr2->getOriginalValue() << " --- " << y[1] << std::endl
                  << "  - constr3_val : "
                  << constr3->getOriginalValue() << " --- " << y[2] << std::endl;

    constr1->setVal(y[0]);
    constr2->setVal(y[1]);
    constr3->setVal(y[2]);

    }
/*----------------------------------------------------------------------------*/

void ConstrBlock3::setFixed(bool fix){
    Constraint* constr;
    for (int i=0;i<m_constr_list->getSize();++i){
        constr=(Constraint*) &((*m_constr_list)[i]);
        constr->setConstant(fix);
        }
    }

/*----------------------------------------------------------------------------*/

/**
Cover all the cases. collapse the problem to a 2x2 or single constraint one.
**/

void ConstrBlock3::m_getBkgConstraints(Constraint** c,
                                       int* n_bkg, 
                                       double* corr){

    // keep trace of what is bkg like here!
    int is_signal[3];

    TString name;
    for (int i=0;i<3;++i){
        name=TString(((*m_constr_list)[i]).GetName());
        is_signal[i]=(int)name.Contains(SIG_KEYWORD);
        std::cout << "DEBUG: is " << name.Data() << " containing " << SIG_KEYWORD
        << "? Answer is " << is_signal[i] << std::endl;
        }


    // Store here the corr coeff to consider in case ith el is sforignal
    int corr_2x2_indeces[3];
    corr_2x2_indeces[0]=2;
    corr_2x2_indeces[1]=1;
    corr_2x2_indeces[2]=0;

    // All are signal 
    if ( (is_signal[0] + is_signal[1] + is_signal[2]) ==3){
        if (is_verbose())
            std::cout << "[m_getBkgConstraints::getBkgNLLstring] "
                      << "All the 3 constraints are for the signal!\n";
        c[0]=c[1]=c[2]=0;
        *n_bkg=0;
        *corr=-1;
        }


    // All are background
    else if ( (is_signal[0] + is_signal[1] + is_signal[2]) ==0){
        if (is_verbose())
            std::cout << "[m_getBkgConstraints::getBkgNLLstring] "
                      << "All the 3 constraints are for the background!\n";

        c[0]=(Constraint*)&((*m_constr_list)[0]);
        c[1]=(Constraint*)&((*m_constr_list)[1]);
        c[2]=(Constraint*)&((*m_constr_list)[2]);

        *n_bkg=3;
        *corr=-1;
        }

    // Two are background
    else if ( (is_signal[0] + is_signal[1] + is_signal[2]) ==1){
        int constr1_index=-1;
        int constr2_index=-1;
        int sign_constr_index=-1;
        if (is_verbose())
            std::cout << "[m_getBkgConstraints::getBkgNLLstring] "
                      << "Two constraints are for background."
                      << " Building the 2 constraints object!\n";
        for (int i=0;i<3;++i){
            if (is_signal[i]==0){
                if (constr1_index==-1)
                    constr1_index=i;
                else
                    constr2_index=i;
                }
            else
                sign_constr_index=i;
            }

        c[constr1_index]=(Constraint*)&((*m_constr_list)[constr1_index]);
        c[constr2_index]=(Constraint*)&((*m_constr_list)[constr2_index]);

        *n_bkg=2;
        *corr=m_corr[corr_2x2_indeces[sign_constr_index]]->getVal();

        if (is_verbose())
            std::cout << "[m_getBkgConstraints::getBkgNLLstring] "
                      << "Constraints for the background:\n"
                      << "  1) " << c[constr1_index]->GetName()
                      << "  2) " << c[constr2_index]->GetName()
                      << std::endl;

        std::cout << "NUMBER of BKG and CORR " << *n_bkg << " " << *corr << "\n";


        }

    // two are for the signal: simple!
    else if ( (is_signal[0] + is_signal[1] + is_signal[2]) ==2){
        if (is_verbose())
            std::cout << "[Constraint::getBkgNLLstring] "
                      << "Only one constraint is for bkg!\n";
        for (int i=0;i<3;++i)
            if (is_signal[i]==0)
                c[i] = (Constraint*)&((*m_constr_list)[i]);

        *n_bkg=1;
        *corr=-1;
        }

    else{
        std::cout << "[m_getBkgConstraints::getBkgNLLstring] "
                  << "All the cases are exhausted.. Aborting!!";
        abort();
        }
    }

/*----------------------------------------------------------------------------*/

ConstrBlock3::~ConstrBlock3(){

    if (m_parameters!=NULL)
        delete m_parameters;
    if (m_constr_list!=NULL)
        delete m_constr_list;

    if (m_Bkg_parameters!=NULL)
        delete m_Bkg_parameters;

    for (int i=0;i<3;++i)
        if (m_corr[i]!=NULL)
            delete m_corr[i];
    }
/*----------------------------------------------------------------------------*/

/// To build the cint dictionaries
//ClassImp(ConstrBlock3)

/*----------------------------------------------------------------------------*/

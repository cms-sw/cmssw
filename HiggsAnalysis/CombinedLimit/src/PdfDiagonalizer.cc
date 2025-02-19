#include "../interface/PdfDiagonalizer.h"

#include <cstdio>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <RooAbsPdf.h>
#include <RooAddition.h>
#include <RooCustomizer.h>
#include <RooFitResult.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

PdfDiagonalizer::PdfDiagonalizer(const char *name, RooWorkspace *w, RooFitResult &result) :
    name_(name),
    parameters_(result.floatParsFinal())
{
    int n = parameters_.getSize();

    TMatrixDSym cov(result.covarianceMatrix()); 
    TMatrixDSymEigen eigen(cov);

    const TMatrixD& vectors = eigen.GetEigenVectors();
    const TVectorD& values  = eigen.GetEigenValues();

    char buff[10240];

    // create unit gaussians per eigen-vector
    for (int i = 0; i < n; ++i) {
        snprintf(buff,sizeof(buff),"%s_eig%d[-5,5]", name, i);
        eigenVars_.add(*w->factory(buff));
    }
    // put them in a list, with a one at the end to set the mean
    RooArgList eigvVarsPlusOne(eigenVars_);
    if (w->var("_one_") == 0) w->factory("_one_[1]");
    eigvVarsPlusOne.add(*w->var("_one_"));

    // then go create the linear combinations
    // each is equal to the transpose matrx times the square root of the eigenvalue (so that we get unit gaussians)
    for (int i = 0; i < n; ++i) {   
        RooArgList coeffs;
        for (int j = 0; j < n; ++j) {
            snprintf(buff,sizeof(buff),"%s_eigCoeff_%d_%d[%g]", name, i, j, vectors(i,j)*sqrt(values(j)));
            coeffs.add(*w->factory(buff)); 
        }
        snprintf(buff,sizeof(buff),"%s_eigBase_%d[%g]", name, i, (dynamic_cast<RooAbsReal*>(parameters_.at(i)))->getVal());
        coeffs.add(*w->factory(buff)); 
        snprintf(buff,sizeof(buff),"%s_eigLin_%d", name, i);
        RooAddition *add = new RooAddition(buff,buff,coeffs,eigvVarsPlusOne);
        w->import(*add);
        replacements_.add(*add);
    }
}

RooAbsPdf *PdfDiagonalizer::diagonalize(RooAbsPdf &pdf)
{
    if (!pdf.dependsOn(parameters_)) return 0;

    // now do the customization
    RooCustomizer custom(pdf, name_.c_str());
    for (int i = 0, n = parameters_.getSize(); i < n; ++i) { 
        if (pdf.dependsOn(*parameters_.at(i))) {
            custom.replaceArg(*parameters_.at(i), *replacements_.at(i));
        }
    }

    RooAbsPdf *ret = dynamic_cast<RooAbsPdf *>(custom.build());
    ret->SetName((std::string(pdf.GetName()) + "_" + name_).c_str());
    return ret;
}

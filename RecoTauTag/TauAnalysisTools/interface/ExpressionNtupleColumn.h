/*
 * ExpressionNtupleColumn
 *
 * Abstract base class which fills the appropriate branch variable
 *
 * ExpressionNtupleColumnT
 *
 * which handles different branch types, I, F, and D.
 *
 * The user should utilize the factory function:
 * std::auto_ptr<ExpressionNtupleColumn<T> > buildColumn<T>(
 *  const std::string& name, const edm::ParameterSet& pset, TTree* tree);
 *
 * Author: Evan K. Friis, UW Madison
 *
 */

#ifndef EXPRESSIONNTUPLECOLUMN_VOU3DWM
#define EXPRESSIONNTUPLECOLUMN_VOU3DWM

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include <TMath.h>

namespace uct {

template<typename ObjType>
class ExpressionNtupleColumn {
  public:
    ExpressionNtupleColumn(const std::string& name, const std::string& func);
    /// Compute the column function and store result in branch variable
    void compute(const ObjType& obj);
    virtual ~ExpressionNtupleColumn() {  }
  protected:
    /// Abstract function which takes the result from the compute and fills the
    /// branch
    virtual void setValue(double value) = 0;
  private:
    std::string name_;
    StringObjectFunction<ObjType> func_;
};

template<typename T>
ExpressionNtupleColumn<T>::ExpressionNtupleColumn(
    const std::string& name, const std::string& func):
  name_(name), func_(func, true) {}

template<typename T> void ExpressionNtupleColumn<T>::compute(const T& obj) {
  this->setValue(func_(obj));
}

namespace {
  template<typename T> std::string getTypeCmd();
  template<> std::string getTypeCmd<Float_t>() { return "/F"; }
  template<> std::string getTypeCmd<Int_t>() { return "/I"; }
  template<> std::string getTypeCmd<Double_t>() { return "/D"; }

  template<typename T> T convertVal(double x);
  template<> Double_t convertVal<Double_t>(double x) { return x; }
  template<> Float_t convertVal<Float_t>(double x) { return x; }
  template<> Int_t convertVal<Int_t>(double x) { return TMath::Nint(x); }
}

// Explicit typed (float, double, etc) ntuple column
template<typename ObjType, typename ColType>
class ExpressionNtupleColumnT : public ExpressionNtupleColumn<ObjType> {
  public:
    ExpressionNtupleColumnT(const std::string& name, const std::string& func,
        TTree* tree);
  protected:
    /// Abstract function
    void setValue(double value);
  private:
    boost::shared_ptr<ColType> branch_;
};

// Explicit typed (float, double, etc) ntuple column
template<typename ObjType, typename ColType>
ExpressionNtupleColumnT<ObjType, ColType>::ExpressionNtupleColumnT(
    const std::string& name, const std::string& func, TTree* tree):
  ExpressionNtupleColumn<ObjType>(name, func) {
    branch_.reset(new ColType);
    std::string branchCmd = name + getTypeCmd<ColType>();
    tree->Branch(name.c_str(), branch_.get(), branchCmd.c_str());
}

template<typename ObjType, typename ColType>
void ExpressionNtupleColumnT<ObjType, ColType>::setValue(double value) {
  *branch_ = convertVal<ColType>(value);
}

template<typename T>
std::auto_ptr<ExpressionNtupleColumn<T> > buildColumn(
    const std::string& name, const edm::ParameterSet& pset, TTree* tree) {

  // The output column
  std::auto_ptr<ExpressionNtupleColumn<T> > output;

  typedef std::vector<std::string> vstring;

  // In the default case (no type specifier) default to float
  if (pset.existsAs<std::string>(name)) {
    output.reset(new ExpressionNtupleColumnT<T, Float_t>(name,
          pset.getParameter<std::string>(name), tree));
  } else if (pset.existsAs<vstring>(name)){
    vstring command = pset.getParameter<vstring>(name);
    if (command.size() != 2) {
      throw cms::Exception("BadTypeSpecifier")
        << "The column " << name << " is declared as a cms.vstring, "
        << " but doesnt not have the required format"
        << " cms.vstring('function', 'type')" << std::endl;
    }
    if (command[1] == "I" || command[1] == "i") {
      // Make a integer column
      output.reset(new ExpressionNtupleColumnT<T, Int_t>(name,
            command[0], tree));
    } else if (command[1] == "F" || command[1] == "f") {
      // Make a float column
      output.reset(new ExpressionNtupleColumnT<T, Int_t>(name,
            command[0], tree));
    } else if (command[1] == "D" || command[1] == "d") {
      // Make a double column
      output.reset(new ExpressionNtupleColumnT<T, Double_t>(name,
            command[0], tree));
    } else {
      throw cms::Exception("BadTypeSpecifier")
        << "The column " << name << " has declared type " << command[1]
        << ", which I don't understand.  Allowed: I, F, and D" << std::endl;
    }
  } else {
      throw cms::Exception("BadColumn")
        << "The column " << name << " is not a cms.vstring or cms.string!"
        << std::endl;
  }
  return output;
}

}

#endif /* end of include guard: EXPRESSIONNTUPLECOLUMN_VOU3DWM */

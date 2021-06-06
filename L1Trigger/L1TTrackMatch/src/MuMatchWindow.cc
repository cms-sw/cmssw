#include "L1Trigger/L1TTrackMatch/interface/MuMatchWindow.h"

MuMatchWindow::MuMatchWindow() { name_ = ""; }

MuMatchWindow::MuMatchWindow(std::string name) { SetName(name); }

MuMatchWindow::~MuMatchWindow() {}

void MuMatchWindow::SetLower(std::string formula) {
  TF1 f("tmp", formula.c_str(), 0, 1000);
  SetLower(&f);
}

void MuMatchWindow::SetCentral(std::string formula) {
  TF1 f("tmp", formula.c_str(), 0, 1000);
  SetCentral(&f);
}

void MuMatchWindow::SetUpper(std::string formula) {
  TF1 f("tmp", formula.c_str(), 0, 1000);
  SetUpper(&f);
}

void MuMatchWindow::SetLower(TF1* formula) {
  if (fLow_)
    throw std::runtime_error("Cannot initialize twice fLow_");
  fLow_ = std::shared_ptr<TF1>((TF1*)formula->Clone((name_ + std::string("low")).c_str()));
}

void MuMatchWindow::SetCentral(TF1* formula) {
  if (fCent_)
    throw std::runtime_error("Cannot initialize twice fCent_");
  fCent_ = std::shared_ptr<TF1>((TF1*)formula->Clone((name_ + std::string("cent")).c_str()));
}

void MuMatchWindow::SetUpper(TF1* formula) {
  if (fHigh_)
    throw std::runtime_error("Cannot initialize twice fHigh_");
  fHigh_ = std::shared_ptr<TF1>((TF1*)formula->Clone((name_ + std::string("high")).c_str()));
}

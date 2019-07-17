#include "L1Trigger/L1TTrackMatch/interface/MuMatchWindow.h"

MuMatchWindow::MuMatchWindow(){
    name_ = "";
}

MuMatchWindow::MuMatchWindow(std::string name){
    SetName(name);
}

MuMatchWindow::~MuMatchWindow(){

}

void MuMatchWindow::SetLower(std::string formula)
{
    // if (f_low)
    //     throw std::runtime_error("Cannot initialize twice f_low");
    // f_low = std::shared_ptr<TF1> (new TF1 ((name_ + std::string("low")).c_str(), formula.c_str(), 0, 1000));
    TF1 f ("tmp", formula.c_str(), 0, 1000);
    SetLower(&f);
}

void  MuMatchWindow::SetUpper(std::string formula)
{
    // if (f_high)
    //     throw std::runtime_error("Cannot initialize twice f_high");
    // f_high = std::shared_ptr<TF1> (new TF1 ((name_ + std::string("high")).c_str(), formula.c_str(), 0, 1000));
    TF1 f ("tmp", formula.c_str(), 0, 1000);
    SetUpper(&f);
}

void MuMatchWindow::SetLower(TF1* formula)
{
    if (f_low)
        throw std::runtime_error("Cannot initialize twice f_low");
    f_low = std::shared_ptr<TF1> ((TF1*) formula->Clone ((name_ + std::string("low")).c_str()));
}

void  MuMatchWindow::SetUpper(TF1* formula)
{
    if (f_high)
        throw std::runtime_error("Cannot initialize twice f_high");
    f_high = std::shared_ptr<TF1> ((TF1*) formula->Clone ((name_ + std::string("high")).c_str()));
}
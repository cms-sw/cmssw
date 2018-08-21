#ifndef MUMATCHWINDOW_H
#define MUMATCHWINDOW_H

#include "TF1.h"
#include <string>
#include <memory>
#include <iostream>

/*
** class  : MuMatchWindow
** author : L.Cadamuro (UF)
** date   : 25/12/2018
** brief  : encodes the lower and upper bounds to match a track to a muon
**          to be flexible, limits are given as strings to create a TF1 function
*/

class MuMatchWindow
{
    public:
        MuMatchWindow();
        MuMatchWindow(std::string name);
        ~MuMatchWindow();
        void SetName(std::string name) {name_ = name;}
        void SetLower(std::string formula);
        void SetUpper(std::string formula);
        void SetLower(TF1* formula);
        void SetUpper(TF1* formula);
        
        // bool matches (double pt);
        double bound_low  (double pt) {return f_low->Eval(pt);}
        double bound_high (double pt) {return f_high->Eval(pt);}


    private:
        std::string name_;
        std::shared_ptr<TF1> f_low;
        std::shared_ptr<TF1> f_high;
};

#endif
// LossFunctions.h
// Here we define the different loss functions that can be used
// with the BDT system. 

#ifndef ADD_LOSS
#define ADD_LOSS

#include "L1Trigger/L1TMuonEndCap/interface/Event.h"
#include <string>
#include <algorithm>

// ========================================================
// ================ Define the Interface ==================
//=========================================================

// Define the Interface
class LossFunction
{
    public:

        // The gradient of the loss function.
        // Each tree is a step in the direction of the gradient
        // towards the minimum of the Loss Function.
        virtual Double_t target(Event* e) = 0;

        // The fit should minimize the loss function in each
        // terminal node at each iteration.
        virtual Double_t fit(std::vector<Event*>& v) = 0;
        virtual std::string name() = 0;
        virtual int id() = 0;
};

// ========================================================
// ================ Least Squares =========================
// ========================================================

class LeastSquares : public LossFunction
{
    public:
        LeastSquares(){}
        ~LeastSquares(){}

        Double_t target(Event* e)
        {
        // Each tree fits the residuals when using LeastSquares.
        return e->trueValue - e->predictedValue;
        }

        Double_t fit(std::vector<Event*>& v)
        {
        // The average of the residuals minmizes the Loss Function for LS.

            Double_t SUM = 0;
            for(unsigned int i=0; i<v.size(); i++)
            {
                Event* e = v[i];
                SUM += e->trueValue - e->predictedValue;
            }
    
            return SUM/v.size();
        }
        std::string name() { return "Least_Squares"; }
        int id(){ return 1; }
       
};

// ========================================================
// ============== Absolute Deviation    ===================
// ========================================================

class AbsoluteDeviation : public LossFunction
{
    public:
        AbsoluteDeviation(){}
        ~AbsoluteDeviation(){}

        Double_t target(Event* e)
        {
        // The gradient.
            if ((e->trueValue - e->predictedValue) >= 0)
                return 1;
            else
                return -1;
        }

        Double_t fit(std::vector<Event*>& v)
        {
        // The median of the residuals minimizes absolute deviation.
            if(v.size()==0) return 0;
            std::vector<Double_t> residuals(v.size());
       
            // Load the residuals into a vector. 
            for(unsigned int i=0; i<v.size(); i++)
            {
                Event* e = v[i];
                residuals[i] = (e->trueValue - e->predictedValue);
            }

            // Get the median and return it.
            int median_loc = (residuals.size()-1)/2;

            // Odd.
            if(residuals.size()%2 != 0)
            {
                std::nth_element(residuals.begin(), residuals.begin()+median_loc, residuals.end());
                return residuals[median_loc];
            }
            
            // Even.
            else
            {
                std::nth_element(residuals.begin(), residuals.begin()+median_loc, residuals.end());
                Double_t low = residuals[median_loc];
                std::nth_element(residuals.begin()+median_loc+1, residuals.begin()+median_loc+1, residuals.end());
                Double_t high = residuals[median_loc+1];
                return (high + low)/2;
            }
        }
        std::string name() { return "Absolute_Deviation"; }
        int id(){ return 2; }
};

// ========================================================
// ============== Huber    ================================
// ========================================================

class Huber : public LossFunction
{
    public:
        Huber(){}
        ~Huber(){}
 
        double quantile;
        double residual_median;

        Double_t target(Event* e)
        {
        // The gradient of the loss function.

            if (TMath::Abs(e->trueValue - e->predictedValue) <= quantile)
                return (e->trueValue - e->predictedValue);
            else
                return quantile*(((e->trueValue - e->predictedValue) > 0)?1.0:-1.0);
        }

        Double_t fit(std::vector<Event*>& v)
        {
        // The constant fit that minimizes Huber in a region.

            quantile = calculateQuantile(v, 0.7);
            residual_median = calculateQuantile(v, 0.5); 

            double x = 0;
            for(unsigned int i=0; i<v.size(); i++)
            {
                Event* e = v[i];
                double residual = e->trueValue - e->predictedValue;
                double diff = residual - residual_median; 
                x += ((diff > 0)?1.0:-1.0)*std::min(quantile, TMath::Abs(diff));
            }

           return (residual_median + x/v.size());
            
        }

        std::string name() { return "Huber"; }
        int id(){ return 3; }

        double calculateQuantile(std::vector<Event*>& v, double whichQuantile)
        {
            // Container for the residuals.
            std::vector<Double_t> residuals(v.size());
       
            // Load the residuals into a vector. 
            for(unsigned int i=0; i<v.size(); i++)
            {
                Event* e = v[i];
                residuals[i] = TMath::Abs(e->trueValue - e->predictedValue);
            }

            std::sort(residuals.begin(), residuals.end());             
            unsigned int quantile_location = whichQuantile*(residuals.size()-1);
            return residuals[quantile_location];
        }        
};

// ========================================================
// ============== Percent Error ===========================
// ========================================================

class PercentErrorSquared : public LossFunction
{
    public:
        PercentErrorSquared(){}
        ~PercentErrorSquared(){}

        Double_t target(Event* e)
        {   
        // The gradient of the squared percent error.
            return (e->trueValue - e->predictedValue)/(e->trueValue * e->trueValue);
        }   

        Double_t fit(std::vector<Event*>& v)
        {   
        // The average of the weighted residuals minimizes the squared percent error.
        // Weight(i) = 1/true(i)^2. 
    
            Double_t SUMtop = 0;
            Double_t SUMbottom = 0;
    
            for(unsigned int i=0; i<v.size(); i++)
            {   
                Event* e = v[i];
                SUMtop += (e->trueValue - e->predictedValue)/(e->trueValue*e->trueValue); 
                SUMbottom += 1/(e->trueValue*e->trueValue);
            }   
    
            return SUMtop/SUMbottom;
        }   
        std::string name() { return "Percent_Error"; }
        int id(){ return 4; }
};

#endif

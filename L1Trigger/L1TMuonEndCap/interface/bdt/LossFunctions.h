// LossFunctions.h
// Here we define the different loss functions that can be used
// with the BDT system.

#ifndef L1Trigger_L1TMuonEndCap_emtf_LossFunctions
#define L1Trigger_L1TMuonEndCap_emtf_LossFunctions

#include "Event.h"
#include <string>
#include <algorithm>
#include <cmath>

// ========================================================
// ================ Define the Interface ==================
//=========================================================

namespace emtf {

// Define the Interface
class LossFunction
{
    public:

        // The gradient of the loss function.
        // Each tree is a step in the direction of the gradient
        // towards the minimum of the Loss Function.
        virtual double target(Event* e) = 0;

        // The fit should minimize the loss function in each
        // terminal node at each iteration.
        virtual double fit(std::vector<Event*>& v) = 0;
        virtual std::string name() = 0;
        virtual int id() = 0;
        virtual ~LossFunction() = default;
};

// ========================================================
// ================ Least Squares =========================
// ========================================================

class LeastSquares : public LossFunction
{
    public:
        LeastSquares(){}
        ~LeastSquares() override{}

        double target(Event* e) override
        {
        // Each tree fits the residuals when using LeastSquares.
        return e->trueValue - e->predictedValue;
        }

        double fit(std::vector<Event*>& v) override
        {
        // The average of the residuals minmizes the Loss Function for LS.

            double SUM = 0;
            for(unsigned int i=0; i<v.size(); i++)
            {
                Event* e = v[i];
                SUM += e->trueValue - e->predictedValue;
            }

            return SUM/v.size();
        }
        std::string name() override { return "Least_Squares"; }
        int id() override{ return 1; }

};

// ========================================================
// ============== Absolute Deviation    ===================
// ========================================================

class AbsoluteDeviation : public LossFunction
{
    public:
        AbsoluteDeviation(){}
        ~AbsoluteDeviation() override{}

        double target(Event* e) override
        {
        // The gradient.
            if ((e->trueValue - e->predictedValue) >= 0)
                return 1;
            else
                return -1;
        }

        double fit(std::vector<Event*>& v) override
        {
        // The median of the residuals minimizes absolute deviation.
            if(v.empty()) return 0;
            std::vector<double> residuals(v.size());

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
                double low = residuals[median_loc];
                std::nth_element(residuals.begin()+median_loc+1, residuals.begin()+median_loc+1, residuals.end());
                double high = residuals[median_loc+1];
                return (high + low)/2;
            }
        }
        std::string name() override { return "Absolute_Deviation"; }
        int id() override{ return 2; }
};

// ========================================================
// ============== Huber    ================================
// ========================================================

class Huber : public LossFunction
{
    public:
        Huber(){}
        ~Huber() override{}

        double quantile;
        double residual_median;

        double target(Event* e) override
        {
        // The gradient of the loss function.

            if (std::abs(e->trueValue - e->predictedValue) <= quantile)
                return (e->trueValue - e->predictedValue);
            else
                return quantile*(((e->trueValue - e->predictedValue) > 0)?1.0:-1.0);
        }

        double fit(std::vector<Event*>& v) override
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
                x += ((diff > 0)?1.0:-1.0)*std::min(quantile, std::abs(diff));
            }

           return (residual_median + x/v.size());

        }

        std::string name() override { return "Huber"; }
        int id() override{ return 3; }

        double calculateQuantile(std::vector<Event*>& v, double whichQuantile)
        {
            // Container for the residuals.
            std::vector<double> residuals(v.size());

            // Load the residuals into a vector.
            for(unsigned int i=0; i<v.size(); i++)
            {
                Event* e = v[i];
                residuals[i] = std::abs(e->trueValue - e->predictedValue);
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
        ~PercentErrorSquared() override{}

        double target(Event* e) override
        {
        // The gradient of the squared percent error.
            return (e->trueValue - e->predictedValue)/(e->trueValue * e->trueValue);
        }

        double fit(std::vector<Event*>& v) override
        {
        // The average of the weighted residuals minimizes the squared percent error.
        // Weight(i) = 1/true(i)^2.

            double SUMtop = 0;
            double SUMbottom = 0;

            for(unsigned int i=0; i<v.size(); i++)
            {
                Event* e = v[i];
                SUMtop += (e->trueValue - e->predictedValue)/(e->trueValue*e->trueValue);
                SUMbottom += 1/(e->trueValue*e->trueValue);
            }

            return SUMtop/SUMbottom;
        }
        std::string name() override { return "Percent_Error"; }
        int id() override{ return 4; }
};

} // end of emtf namespace

#endif

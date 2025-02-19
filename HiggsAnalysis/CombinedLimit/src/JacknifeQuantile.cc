#include "../interface/JacknifeQuantile.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <Math/QuantFuncMathMore.h>
#include <RooAbsData.h>
#include <RooRealVar.h>

QuantileCalculator::QuantileCalculator()
{
}

QuantileCalculator::~QuantileCalculator()
{
}

QuantileCalculator::QuantileCalculator(const std::vector<double> &values, const std::vector<double> &weights)
{
   import<double>(values, weights); 
}

QuantileCalculator::QuantileCalculator(const std::vector<float> &values, const std::vector<float> &weights)
{
   import<float>(values, weights); 
}

QuantileCalculator::QuantileCalculator(const RooAbsData &data, const char *varName, int firstEntry, int lastEntry)
{
    if (lastEntry == -1) lastEntry = data.numEntries();
    const RooArgSet  *set = data.get();
    const RooRealVar *x   = dynamic_cast<const RooRealVar *>(set->find(varName)); 
    if (x == 0) {
     set->Print("V");
     throw std::logic_error("Parameter of interest not in the idataset");
    }
    sumw_.resize(1, 0.0);
    if (firstEntry < lastEntry) {
        points_.resize(lastEntry - firstEntry);
        for (int i = firstEntry, j = 0; i < lastEntry; ++i, ++j) {
            data.get(i); 
            points_[j].x = x->getVal(); 
            points_[j].w = data.weight();
            sumw_[0] += points_[j].w;
        }
    }
}

void QuantileCalculator::randomizePoints()
{
    std::random_shuffle(points_.begin(), points_.end());
}

std::pair<double,double> QuantileCalculator::quantileAndError(double quantile, Method method) 
{
    if (method == Simple) {
        std::sort(points_.begin(), points_.end());
        quantiles(quantile, false);
        return std::pair<double,double>(quantiles_[0], 0);
    } else if (method == Sectioning || method == Jacknife) {
        int m = guessPartitions(points_.size(), quantile);
        partition(m, (method == Jacknife));
        std::sort(points_.begin(), points_.end());
        quantiles(quantile, (method == Jacknife));
        double avg = 0;
        for (int i = 0; i < m; ++i) {
            avg += quantiles_[i];
        }
        avg /= m;
        double rms = 0;
        for (int i = 0; i < m; ++i) {
            rms += (quantiles_[i] - avg)*(quantiles_[i] - avg);
        }
        rms = sqrt(rms/(m*(m-1)));
        double onesigma = ROOT::Math::tdistribution_quantile_c(0.16, m-1);
        return std::pair<double,double>(avg, rms * onesigma);
    }
    return std::pair<double,double>(0,-1);
}

int QuantileCalculator::guessPartitions(int size, double quantile) 
{
    // number of events on the small side of the quantile for m=1
    double smallnum = size * std::min(quantile, 1-quantile); 
    // now the naive idea is that err(q) ~ 1/sqrt(smallnum/m), while the error from averaging does as ~1/sqrt(m)
    // so you want m ~ sqrt(smallnum)
    int n = 5; //std::min(std::max(3., sqrt(smallnum)), 5.);
    std::cout << "   .....  will split the " << size << " events in " << n << " subsets, smallnum is " << smallnum << std::endl;
    return n;
}

template<typename T>
void QuantileCalculator::import(const std::vector<T> &values, const std::vector<T> &weights) 
{
    int n = values.size();
    points_.resize(values.size());
    sumw_.resize(1); sumw_[0] = 0.0;
    for (int i = 0; i < n; ++i) {
        points_[i].x = values[i];
        points_[i].w = weights.empty() ? 1 : weights[i];
        points_[i].set = 0;
        sumw_[0] += points_[i].w;
    }
}


void QuantileCalculator::partition(int m, bool doJacknife) 
{
    int n = points_.size();
    sumw_.resize(m+1);
    std::fill(sumw_.begin(), sumw_.end(), 0.0);
    double alpha = double(m)/n;
    for (int i = 0, j = 0; i < n; ++i) {
        j = floor(i*alpha); 
        points_[i].set = j;
        sumw_[j] += points_[i].w;
    }
    if (doJacknife) {
        // at this point sumw[j] has the weights of j... 
        // now I have to get the weights of everyhing else
        // start with all weights
        for (int j = 0; j < m; ++j) sumw_[m] += sumw_[j];
        // and then subtract
        for (int j = 0; j < m; ++j) sumw_[j] = (sumw_[m] - sumw_[j]);
    }
}

void QuantileCalculator::quantiles(double quantile, bool doJacknife) 
{
    int m = sumw_.size()-1;
    int n = points_.size();
    quantiles_.resize(m+1);
    for (int j = 0; j <= m; ++j) {
        double runningSum = 0;
        double threshold  = quantile * sumw_[j];
        int ilow = 0, ihigh = n-1;
        for (int i = 0; i < n; ++i) {
            if ((points_[i].set == j) == doJacknife) continue; // if jacknife, cut away just one piece, otherwise cut away everthing else
            //std::cout << "\t\t\t" << points_[i].x << std::endl;;
            if (runningSum + points_[i].w <= threshold) { 
                runningSum += points_[i].w;
                ilow = i;
            } else {
                ihigh = i;
                break;
            }
        }
        if (runningSum == threshold) { // possible if all unit weights
            quantiles_[j] = points_[ilow].x;
        } else {
            quantiles_[j] = 0.5*(points_[ilow].x + points_[ihigh].x);
        }
    }
    for (int j = 0; j <= m; ++j) {  
        //printf("   ... quantile of section %d: %6.3f\n", j, quantiles_[j]);
    }
    if (doJacknife) {
        // now compute the pseudo-values alpha_[j] = m * quantile[m] - (m-1) * quantile[j]
        for (int j = 0; j < m; ++j) {
            quantiles_[j] = m * quantiles_[m] - (m-1) * quantiles_[j];
            printf("   ... jacknife quantile of section %d: %6.3f\n", j, quantiles_[j]);
        }
    }
}

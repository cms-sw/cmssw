#include "../interface/FastTemplate.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

FastTemplate::T FastTemplate::Integral() const {
    T total = 0;
    for (unsigned int i = 0; i < size_; ++i) total += values_[i];
    return total;
}

void FastTemplate::Scale(T factor) {
    for (unsigned int i = 0; i < size_; ++i) values_[i] *= factor;
}

void FastTemplate::Clear() {
    for (unsigned int i = 0; i < size_; ++i) values_[i] = T(0.);
}

void FastTemplate::CopyValues(const FastTemplate &other) {
    memcpy(values_, other.values_, size_*sizeof(T));
}

void FastTemplate::CopyValues(const TH1 &other) {
     for (unsigned int i = 0; i < size_; ++i) values_[i] = other.GetBinContent(i+1);
}

void FastTemplate::Dump() {
    printf("--- dumping template with %d bins (@%p) ---\n", size_+1, (void*)values_);
    for (unsigned int i = 0; i < size_; ++i) printf(" bin %3d: yval = %9.5f\n", i, values_[i]);
    printf("\n"); 
}

FastHisto::FastHisto(const TH1 &hist) :
    FastTemplate(hist),
    binEdges_(new T[size_+1]),
    binWidths_(new T[size_])
{
    for (unsigned int i = 0; i < size_; ++i) {
        binEdges_[i] = hist.GetBinLowEdge(i+1);
        binWidths_[i] = hist.GetBinWidth(i+1);
    }
    binEdges_[size_] = hist.GetBinLowEdge(size_+1);
}

FastHisto::FastHisto(const FastHisto &other) :
    FastTemplate(other),
    binEdges_(new T[size_+1]),
    binWidths_(new T[size_])
{
    memcpy(binEdges_,  other.binEdges_, (size_+1)*sizeof(T));
    memcpy(binWidths_, other.binWidths_, size_*sizeof(T));
}

FastHisto::T FastHisto::GetAt(const T &x) const {
    T *match = std::lower_bound(binEdges_, binEdges_+size_+1, x);
    if (match == binEdges_ || match == binEdges_+size_+1) return T(0.0);
    return values_[match - binEdges_ - 1];
}

FastHisto::T FastHisto::IntegralWidth() const {
    T total = 0;
    for (unsigned int i = 0; i < size_; ++i) total += values_[i] * binWidths_[i];
    return total;
}

void FastHisto::Dump() {
    printf("--- dumping histo template with %d bins in range %.2f - %.2f (@%p)---\n", size_+1, binEdges_[0], binEdges_[size_], (void*)values_);
    for (unsigned int i = 0; i < size_; ++i) {
        printf(" bin %3d, x = %6.2f: yval = %9.5f, width = %6.3f\n", 
                    i, 0.5*(binEdges_[i]+binEdges_[i+1]), values_[i], binWidths_[i]);
    }
    printf("\n"); 
}



namespace { 
    /// need the __restrict__ to make them work 
    void subtract(FastTemplate::T * __restrict__ out, unsigned int n, FastTemplate::T  const * __restrict__ ref) {
        for (unsigned int i = 0; i < n; ++i) out[i] -= ref[i];
    }
    void logratio(FastTemplate::T * __restrict__ out, unsigned int n, FastTemplate::T  const * __restrict__ ref) {
        for (unsigned int i = 0; i < n; ++i) {
            out[i] = (out[i] > 0 && ref[i] > 0) ? std::log(out[i]/ref[i]) : 0;
        }
    }
    void sumdiff(FastTemplate::T * __restrict__ sum, FastTemplate::T * __restrict__ diff, 
                 unsigned int n, 
                 const FastTemplate::T  * __restrict__ h1, const FastTemplate::T  * __restrict__ h2) {
        //printf("sumdiff(sum = %p, diff = %p, n = %d, h1 = %p, h2 = %p\n", (void*)sum, (void*)diff, n, (void*)h1, (void*)h2);
        for (unsigned int i = 0; i < n; ++i) {
            sum[i]  = h1[i] + h2[i];
            diff[i] = h1[i] - h2[i];
            //printf("%3d: sum = %.6f, diff = %.6f, h1 = %.6f, h2 = %.6f\n", i, sum[i], diff[i], h1[i], h2[i]);
        }
    }
    void meld(FastTemplate::T * __restrict__ out, unsigned int n, FastTemplate::T  const * __restrict__ diff, FastTemplate::T  const * __restrict__ sum, FastTemplate::T x, FastTemplate::T y) {
        for (unsigned int i = 0; i < n; ++i) {
            out[i] += x*(diff[i] + y*sum[i]);
        }
    }
}

void FastTemplate::Subtract(const FastTemplate & ref) {
    subtract(values_, size_, &ref[0]);
}
void FastTemplate::LogRatio(const FastTemplate & ref) {
    logratio(values_, size_, &ref[0]);
}
void FastTemplate::SumDiff(const FastTemplate & h1, const FastTemplate & h2, 
                           FastTemplate & sum, FastTemplate & diff) {
    sumdiff(&sum[0], &diff[0], h1.size_, &h1[0], &h2[0]);
}

void FastTemplate::Meld(const FastTemplate & diff, const FastTemplate & sum, T x, T y) {
    meld(values_, size_, &diff[0], &sum[0], x, y);
}

void FastTemplate::Log() {
    for (unsigned int i = 0; i < size_; ++i) {
        values_[i] = values_[i] > 0 ? std::log(values_[i]) : T(-999);
    }
}

void FastTemplate::Exp() {
    for (unsigned int i = 0; i < size_; ++i) {
        values_[i] = std::exp(values_[i]);
    }
}

void FastTemplate::CropUnderflows(T minimum) {
    for (unsigned int i = 0; i < size_; ++i) {
        if (values_[i] < minimum) values_[i] = minimum;
    }
}   

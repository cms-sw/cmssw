#ifndef HiggsAnalysis_CombinedLimit_FastH1
#define HiggsAnalysis_CombinedLimit_FastH1

#include <TH1.h>
#include <TH2.h>
#include <algorithm>

class FastTemplate {
    public:
        typedef float T;
        FastTemplate() : size_(0), values_(0) {}
        FastTemplate(unsigned int size) : size_(size), values_(new T[size_]) {}
        FastTemplate(const FastTemplate &other) : size_(other.size_), values_(new T[size_]) { CopyValues(other); }
        FastTemplate(const TH1 &other) : size_(other.GetNbinsX()), values_(new T[size_]) { CopyValues(other); }
        FastTemplate(const TH2 &other) : size_(other.GetNbinsX()*other.GetNbinsY()), values_(new T[size_]) { CopyValues(other); }
        FastTemplate & operator=(const FastTemplate &other) { 
            if (size_ != other.size_) { 
                delete [] values_; size_ = other.size_; values_ = new T[size_];
            }
            CopyValues(other); return *this; 
        }
        FastTemplate & operator=(const TH1 &other) { 
            if (size_ != unsigned(other.GetNbinsX())) { 
                delete [] values_; size_ = other.GetNbinsX(); values_ = new T[size_];
            }
            CopyValues(other); return *this;  
        }
        ~FastTemplate() { delete [] values_; }
        void Resize(unsigned int newsize) {
            if (newsize != size_) {
                delete [] values_; size_ = newsize; values_ = new T[size_];
            }
        }
        T Integral() const ;
        void Scale(T factor) ;
        void Clear() ; 
        void CopyValues(const FastTemplate &other) ;
        void CopyValues(const TH1 &other) ;
        void CopyValues(const TH2 &other) ;
        T & operator[](unsigned int i) { return values_[i]; }
        const T & operator[](unsigned int i) const { return values_[i]; }
        const unsigned int size() const { return size_; }
        
        /// *this = log(*this) 
        void Log();
        /// *this = exp(*this) 
        void Exp();
        /// *this = *this - reference
        void Subtract(const FastTemplate &reference);
        /// *this = log(*this)/(reference)
        void LogRatio(const FastTemplate &reference);
        /// assigns sum and diff
        static void SumDiff(const FastTemplate &h1, const FastTemplate &h2, FastTemplate &sum, FastTemplate &diff);
        /// Does this += x * (diff + (sum)*y)
        void Meld(const FastTemplate & diff, const FastTemplate & sum, T x, T y) ;
        /// protect from underflows (*this = max(*this, minimum));
        void CropUnderflows(T minimum=1e-9);

        void Dump() const ;
    protected:
        unsigned int size_; 
        T *values_;
};
class FastHisto : public FastTemplate {
    public:
        FastHisto() : FastTemplate(), binEdges_(0), binWidths_(0) {}
        FastHisto(const TH1 &hist) ;
        FastHisto(const FastHisto &other) ;
        FastHisto & operator=(const FastHisto &other) { 
            if (size_ != other.size_) {
                FastHisto fh(other);
                swap(fh);
            } else CopyValues(other); 
            return *this; 
        }
        FastHisto & operator=(const TH1 &other) { 
            if (size_ != unsigned(other.GetNbinsX())) { 
                FastHisto fh(other);
                swap(fh);
            } else CopyValues(other); 
            CopyValues(other); return *this;  
        }
        ~FastHisto() { delete [] binEdges_; delete [] binWidths_; }
        void swap(FastHisto &other) {
            std::swap(size_, other.size_);
            std::swap(values_, other.values_);
            std::swap(binWidths_, other.binWidths_);
            std::swap(binEdges_, other.binEdges_);
        }
        T GetAt(const T &x) const ;
        T IntegralWidth() const ;
        void Normalize() {
            T sum = IntegralWidth();
            if (sum > 0) Scale(1.0f/sum);
        }

        void Dump() const ;
    private:
        T *binEdges_;
        T *binWidths_;
    
};
class FastHisto2D : public FastTemplate {
    public:
        FastHisto2D() : FastTemplate(), binX_(0), binY_(0), binEdgesX_(0), binEdgesY_(0), binWidths_(0) {}
        FastHisto2D(const TH2 &hist, bool normXonly=false) ;
        FastHisto2D(const FastHisto2D &other) ;
        FastHisto2D & operator=(const FastHisto2D &other) { 
            if (binX_ != other.binY_ || binY_ != other.binY_) {
                FastHisto2D fh(other);
                swap(fh);
            } else CopyValues(other); 
            return *this; 
        }
        ~FastHisto2D() { delete [] binEdgesX_; delete [] binEdgesY_; delete [] binWidths_; }
        void swap(FastHisto2D &other) {
            std::swap(binX_, other.binX_);
            std::swap(binY_, other.binY_);
            std::swap(size_, other.size_);
            std::swap(values_, other.values_);
            std::swap(binWidths_, other.binWidths_);
            std::swap(binEdgesX_, other.binEdgesX_);
            std::swap(binEdgesY_, other.binEdgesY_);
        }
        T GetAt(const T &x, const T &y) const ;
        T IntegralWidth() const ;
        void Normalize() {
            T sum = IntegralWidth();
            if (sum > 0) Scale(1.0f/sum);
        }
        /// For each X, normalize along Y
        void NormalizeXSlices() ;

        void Dump() const ;
    private:
        unsigned int binX_, binY_;
        T *binEdgesX_;
        T *binEdgesY_;
        T *binWidths_;
    
};

#endif

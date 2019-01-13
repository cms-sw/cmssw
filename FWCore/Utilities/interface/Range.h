#ifndef FWCore_Utilities_Range_h
#define FWCore_Utilities_Range_h

namespace edm
{
     /*
      *class which implements begin() and end() to use range-based loop with
      pairs of iterators or pointers.
      */

     template<class T>
     class Range
     {
       public:
         Range(T begin, T end) : begin_(begin), end_(end) {}

         T begin() const { return begin_; }
         T end() const { return end_; }

         bool empty() const { return begin_ == end_; }

       private:
         const T begin_;
         const T end_;
     };
 };

#endif

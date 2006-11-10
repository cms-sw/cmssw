#ifndef DDI_rep_type_h
#define DDI_rep_type_h

namespace DDI {

 template <class N, class I> 
 struct rep_traits
 {
   typedef N name_type;
   typedef typename I::value_type value_type;
   typedef typename I::pointer pointer;
   typedef typename I::reference reference;
 };
 
 template <class N, class I> 
 struct rep_traits<N,I*>
 {
   typedef N name_type;
   typedef I value_type;
   typedef I* pointer;
   typedef I& reference;
 };
 
 
 template <class N, class I>
 struct rep_type
 {
   rep_type() : second(0), init_(false) {}
   rep_type(const N & n, I i) : first(n), second(i), init_(false) 
    { if (i) init_=true; }
   virtual ~rep_type(){}
   N first;
   I second;
   bool init_;
   const typename rep_traits<N,I>::name_type & name() const { return first; }
   const typename rep_traits<N,I>::reference rep()  const { return *second; }
         typename rep_traits<N,I>::reference rep()        { return *second; }
   I swap(I i) { I tmp(second); 
                 second = i; 
		 init_ = false;
		 if (i) init_ = true;
		 return tmp; 
	        }	 
   operator bool() const { return init_; }	 
 };

}
#endif

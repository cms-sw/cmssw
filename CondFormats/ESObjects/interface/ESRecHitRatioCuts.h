#ifndef CondFormats_ESObjects_ESRecHitRatioCuts_H
#define CondFormats_ESObjects_ESRecHitRatioCuts_H
#include <iostream>

class ESRecHitRatioCuts {

  public:

    ESRecHitRatioCuts();
    ESRecHitRatioCuts(const float & r12Low, const float & r23Low, 
		      const float & r12High, const float & r23High);
    ~ESRecHitRatioCuts();

    void  setR12Low(const float& value) { r12Low_ = value; }
    float getR12Low() const { return r12Low_; }
    void  setR23Low(const float& value) { r23Low_ = value; }
    float getR23Low() const { return r23Low_; }

    void  setR12High(const float& value) { r12High_ = value; }
    float getR12High() const { return r12High_; }
    void  setR23High(const float& value) { r23High_ = value; }
    float getR23High() const { return r23High_; }

    void print(std::ostream& s) const {
      s << "ESRecHitRatioCuts: ES low cut (r12, r23) / high cut (r12, r23)" << r12Low_ << " " << r23Low_<< " / " << r12High_ <<" "<< r23High_;
    }

  private:

    float r12Low_;
    float r23Low_;
    float r12High_;
    float r23High_;
};

#endif

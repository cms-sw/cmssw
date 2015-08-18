#ifndef DataFormats_ScoutingVertex_h
#define DataFormats_ScoutingVertex_h

#include <vector>

//class for holding vertex information, for use in data scouting 
//IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class ScoutingVertex
{
    public:
        //constructor with values for all data fields
        ScoutingVertex(float x, float y, float z, float zError):
            x_(x), y_(y), z_(z), zError_(zError) {}
        //default constructor
        ScoutingVertex(): x_(0), y_(0), z_(0), zError_(0) {}

        //accessor functions
        float x() const { return x_; }
        float y() const { return y_; }
        float z() const { return z_; }
        float zError() const { return zError_; }

    private:
        float x_;
        float y_;
        float z_;
        float zError_;
};

typedef std::vector<ScoutingVertex> ScoutingVertexCollection;

#endif

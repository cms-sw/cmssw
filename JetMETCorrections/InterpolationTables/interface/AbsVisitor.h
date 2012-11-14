#ifndef NPSTAT_ABSVISITOR_HH_
#define NPSTAT_ABSVISITOR_HH_

/*!
// \file AbsVisitor.h
//
// \brief Interface for piecemeal processing of a data collection
//
// Author: I. Volobouev
//
// March 2010
*/

namespace npstat {
    /**
    // Interface class for piecemeal processing of a data collection
    */
    template <typename Input, typename Result>
    struct AbsVisitor
    {
        inline virtual ~AbsVisitor() {}

        /** Clear all accumulated results */
        virtual void clear() = 0;

        /** Process one array point */
        virtual void process(const Input& value) = 0;

        /** Return the result at the end of array processing */
        virtual Result result() = 0;
    };

    /**
    // Simple counter of visits is needed often, so it makes sense
    // to put it together with AbsVisitor in the same header. Do not
    // derive from this class, its destructor is not virtual.
    */
    template <typename Input>
    class VisitCounter : public AbsVisitor<Input,unsigned long>
    {
    public:
        inline VisitCounter() : counter_(0UL) {}

        inline void clear() {counter_ = 0UL;}
        inline void process(const Input&) {++counter_;}
        inline unsigned long result() {return counter_;}

    private:
        unsigned long counter_;
    };
}

#endif // ABSVISITOR_HH_


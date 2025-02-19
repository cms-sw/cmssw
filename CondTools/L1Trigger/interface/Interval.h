#ifndef CondTools_L1Trigger_Interval_h
#define CondTools_L1Trigger_Interval_h

#include <map>
#include <cassert>

namespace l1t
{
    /* Template class that will be used to represnt interval from one value to another.
     * In general this class is not interested in what interval mark means, most of the time
     * it should be number, time or something similar
     *
     * This class requires that TimeType should have defined operator == and < as defined in STL.
     * It is enforced via sserts that start time is less then end time under provided operator <.
     * TimeType requires empty constructor.
     *
     * Payload should have defined copy constructor and assigment operator.
     */
    template<typename TimeType, typename PayloadType>
    class Interval
    {
    public:
        /* Construncts the class with provided start and end times. Payload is created
         * with default constructor.
         */
        Interval (const TimeType& start, const TimeType& end)
            : m_start (start), m_end (end), isInvalid (false)
        { assert (m_start <= m_end); }

        /* Constructs the class with given start and end times, as well as given payload.
         */
        Interval (const TimeType& start, const TimeType& end, const PayloadType& payload)
            : m_start(start), m_end (end), _payload (payload), isInvalid (false) {}

        /* Sets the payload to the given one. */
        void setPayload (const PayloadType& payload) { this->_payload = payload; }
        /* Returns the payload */
        const PayloadType& payload () const { return this->_payload; }

        /* Returns start time */
        const TimeType & start () const { return this->m_start; }
        /* Returns end time */
        const TimeType & end () const { return this->m_end; }

        /* Static member that will define an invalid interval. Two invalid interfaces are
         * always considered equal.
         */
        static Interval & invalid ();

        // Operator overloading
        bool operator== (const Interval<TimeType, PayloadType> & other) const
        { return (this->isInvalid == true && other.isInvalid == true ) ||
            (this->start () == other.start ()) && (this->end () == other.end () &&
                    this->isInvalid == other.isInvalid); }

        bool operator!= (const Interval<TimeType, PayloadType> & other) const
        { return ! (*this == other); }

    protected:
        /* Private data */
        TimeType m_start;
        TimeType m_end;
        PayloadType _payload;

        /* flag that will check if this interval is invalid */
        bool isInvalid;
    };

    /* Manages a list of intervals and provides method to find interval that contains
     * some value.
     *
     * Template parameters are used to manage Interval class, so all requirements to these
     * parameters comes from Interval class
     */
    template<typename TimeType, typename PayloadType>
    class IntervalManager
    {
    public:
        /* Adds one given interval to the list of intervals.
         */
        void addInterval (const Interval<TimeType, PayloadType> & interval)
        { intervalMap.insert (std::make_pair (interval.start (), interval)); }

        /* Removes all stored intervals from the list
         */
        void clear () { intervalMap.clear (); }

        /* Returns interval that contaisn given time. If multiple intervals exists
         * any of them is returned
         */
        const Interval<TimeType, PayloadType> & find (const TimeType & time) const;

    protected:
        /* Information to store list of intervals */
        typedef std::map<TimeType, Interval<TimeType, PayloadType> > IntervalMap;
        IntervalMap intervalMap;
    };

} // namespace

// implementation
#include "CondTools/L1Trigger/src/Interval.icc"

#endif

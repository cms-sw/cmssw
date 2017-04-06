#ifndef CondFormats_RPCObjects_RPCDCCLink_h
#define CondFormats_RPCObjects_RPCDCCLink_h

#include <stdint.h>
#include <string>
#include <iosfwd>
#include <climits>

#include "CondFormats/Serialization/interface/Serializable.h"

/** Identifier for RPC DCC-LB path */
class RPCDCCLink
{
public:
    static int const wildcard_ = INT_MIN;

    /** @{ */
    /** field ranges */
    static int const min_fed_ = 0;
    static int const max_fed_ = 65534;
    static int const min_dccinput_ = 0;
    static int const max_dccinput_ = 36;
    static int const min_tbinput_ = 0;
    static int const max_tbinput_ = 18;
    /** @} */

protected:
    /** @{ */
    /** field positions and masks */
    static int const pos_fed_ = 16;
    static ::uint32_t const mask_fed_ = 0xffff0000;
    static int const pos_dccinput_ = 8;
    static ::uint32_t const mask_dccinput_ = 0x0000ff00;
    static int const pos_tbinput_ = 0;
    static ::uint32_t const mask_tbinput_ = 0x000000ff;
    /** @} */

public:
    RPCDCCLink();
    RPCDCCLink(::uint32_t const & _id);
    RPCDCCLink(int _fed
               , int _dccinput
               , int _tbinput = wildcard_);

    ::uint32_t getId() const;
    operator ::uint32_t() const;
    ::uint32_t getMask() const;

    bool matches(RPCDCCLink const & _rhs) const;

    void setId(::uint32_t const & _id);
    void reset();

    /** @{ */
    /** Field Getters */
    int getFED() const;
    int getDCCInput() const;
    int getTBInput() const;
    /** @} */

    /** @{ */
    /** Field Setters
     * A cms::Exception("OutOfRange") is thrown for out-of-range input values.
     **/
    RPCDCCLink & setFED(int _fed = wildcard_);
    RPCDCCLink & setDCCInput(int _dccinput = wildcard_);
    RPCDCCLink & setTBInput(int _tbinput = wildcard_);
    /** @} */

    std::string getName() const;

    bool operator<(RPCDCCLink const & _rhs) const;
    bool operator==(RPCDCCLink const & _rhs) const;
    bool operator!=(RPCDCCLink const & _rhs) const;
    bool operator<(::uint32_t const & _rhs) const;
    bool operator==(::uint32_t const & _rhs) const;
    bool operator!=(::uint32_t const & _rhs) const;

    RPCDCCLink & operator++();
    RPCDCCLink operator++(int);
    RPCDCCLink & operator--();
    RPCDCCLink operator--(int);

protected:
    int bf_get(int const _min, ::uint32_t const _mask, int const _pos) const;
    RPCDCCLink & bf_set(int const _min, int const _max, ::uint32_t const _mask, int const _pos, int const _value);
    std::ostream & bf_stream(std::ostream & _ostream, int const _min, ::uint32_t const _mask, int const _pos) const;

protected:
    ::uint32_t id_;

    COND_SERIALIZABLE;
};

std::ostream & operator<<(std::ostream & _ostream, RPCDCCLink const & _link);

#include "CondFormats/RPCObjects/interface/RPCDCCLink.icc"

#endif // CondFormats_RPCObjects_RPCDCCLink_h

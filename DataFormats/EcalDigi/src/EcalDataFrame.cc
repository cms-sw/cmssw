#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

int EcalDataFrame::lastUnsaturatedSample() const
{
        int cnt = 0;
        for ( size_t i = 3; i < m_data.size(); ++i ) {
                cnt = 0;
                for ( size_t j = i; j < (i + 5) && j < m_data.size(); ++j ) {
                        if ( ((EcalMGPASample)m_data[j]).gainId() == 0 ) ++cnt;
                }
                if ( cnt == 5 ) return i-1; // the last unsaturated sample
        }
        return -1; // no saturation found
}

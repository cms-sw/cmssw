#include <iostream>

#include "TH1F.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"



FWMagField::FWMagField() :
   TEveMagField(),

   m_source(kNone),
   m_userField(-1),
   m_eventField(-1),

   m_reverse(true),
   m_simpleModel(false),

   m_guessValHist(0),
   m_numberOfFieldIsOnEstimates(0),
   m_numberOfFieldEstimates(0),
   m_updateFieldEstimate(true)
{
   m_guessValHist = new TH1F("FieldEstimations", "Field estimations from tracks and muons",
                             200, -4.5, 4.5);
   m_guessValHist->SetDirectory(0);
}

FWMagField::~FWMagField()
{
   delete m_guessValHist;
}

//______________________________________________________________________________

TEveVector
FWMagField::GetField(Float_t x, Float_t y, Float_t z) const
{
   // Virtual method of TEveMagField class.

   Float_t R = sqrt(x*x+y*y);
   Float_t field = m_reverse ? -GetMaxFieldMag() : GetMaxFieldMag();

   //barrel
   if ( TMath::Abs(z)<724 ){
    
      //inside solenoid
      if ( R < 300) return TEveVector(0,0,field);
      // outside solinoid
      if ( m_simpleModel ||
           ( R>461.0 && R<490.5 ) ||
           ( R>534.5 && R<597.5 ) ||
           ( R>637.0 && R<700.0 ) )
         return TEveVector(0,0,-field/3.8*1.2);
  
   } else {
      // endcaps
      if (m_simpleModel){
         if ( R < 50 ) return TEveVector(0,0,field);
         if ( z > 0 )
            return TEveVector(x/R*field/3.8*2.0, y/R*field/3.8*2.0, 0);
         else
            return TEveVector(-x/R*field/3.8*2.0, -y/R*field/3.8*2.0, 0);
      }
      // proper model
      if ( ( ( TMath::Abs(z)>724 ) && ( TMath::Abs(z)<786 ) ) ||
           ( ( TMath::Abs(z)>850 ) && ( TMath::Abs(z)<910 ) ) ||
           ( ( TMath::Abs(z)>975 ) && ( TMath::Abs(z)<1003 ) ) )
      {
         if ( z > 0 )
            return TEveVector(x/R*field/3.8*2.0, y/R*field/3.8*2.0, 0);
         else
            return TEveVector(-x/R*field/3.8*2.0, -y/R*field/3.8*2.0, 0);
      }
   }
   return TEveVector(0,0,0);
}

//______________________________________________________________________________

Float_t
FWMagField::GetMaxFieldMag() const
{
   float res;
   switch ( m_source )
   {
      case kEvent:
      {
         res = m_eventField;  
         break;
      }
      case kUser:
      {
         res = m_userField;   
         break;
      }
      default:
      {
         if ( m_updateFieldEstimate )
         {
            if ( m_guessValHist->GetEntries() > 2  && m_guessValHist->GetRMS()  < 0.5 )
            {
               m_guessedField = m_guessValHist->GetMean();
            

               // std::cout << "FWMagField::GetMaxFieldMag(), get average "
               //  << m_guessValHist->GetMean() << " guessed value: RMS= "<< m_guessValHist->GetRMS()
               //  <<" samples "<< m_guessValHist->GetEntries() << std::endl;
            
            }
            else if ( m_numberOfFieldIsOnEstimates > m_numberOfFieldEstimates/2 || m_numberOfFieldEstimates == 0 )
            {
               m_guessedField = 3.8;
               // fwLog(fwlog::kDebug) << "FWMagField::GetMaxFieldMag() get default field, number estimates "
               //  << m_numberOfFieldEstimates << " number fields is on  m_numberOfFieldIsOnEstimates" <<std::endl;
            }
            else
            {
               m_guessedField = 0;
               //  fwLog(fwlog::kDebug) << "Update field estimate, guess field is OFF." <<std::endl;
            }
            m_updateFieldEstimate  = false;
         }

         res = m_guessedField;
      }
   }

   return res;
}

//______________________________________________________________________________

void FWMagField::guessFieldIsOn(bool isOn) const
{
   if ( isOn ) ++m_numberOfFieldIsOnEstimates;
   ++m_numberOfFieldEstimates;
   m_updateFieldEstimate  = true;
}

void FWMagField::guessField(float val) const
{
   // fwLog(filedDebug) <<  "FWMagField::guessField "<< val << std::endl;
   m_guessValHist->Fill(val);
   m_updateFieldEstimate = true; 
}

void FWMagField::resetFieldEstimate() const
{
   m_guessValHist->Reset();
   m_guessValHist->SetAxisRange(-4, 4);
   m_numberOfFieldIsOnEstimates = 0;
   m_numberOfFieldEstimates = 0;
   m_updateFieldEstimate = true;   
}

//______________________________________________________________________________
void FWMagField::checkFiledInfo(const fwlite::Event* event)
{
   const static float  currentToFiled = 3.8/18160;
   bool available = false;
   try
   {
      fwlite::Handle<edm::ConditionsInRunBlock> runCond;
      runCond.getByLabel( event->getRun(), "conditionsInEdm","", "" );
      if( runCond.isValid())
      {
         available = true;
         m_eventField = currentToFiled * runCond->BAvgCurrent;
         fwLog( fwlog::kDebug ) << "Magnetic field info found in ConditionsInEdm branch : "<< m_eventField << std::endl;
      }
      else
      {
         fwlite::Handle< std::vector<DcsStatus> > dcsStatus;
         dcsStatus.getByLabel( *event, "scalersRawToDigi", "", "" );
         if (dcsStatus.isValid() && dcsStatus->size())
         {
            float sum = 0;
            for (std::vector<DcsStatus>::const_iterator i = dcsStatus->begin(); i <  dcsStatus->end(); ++i)
               sum += (*i).magnetCurrent();

            available = true;
            m_eventField = currentToFiled * sum/dcsStatus->size();
            fwLog( fwlog::kDebug) << "Magnetic field info found in DcsStatus branch: " << m_eventField << std::endl;
         }

      }
   }
   catch ( cms::Exception&)
   {
      fwLog( fwlog::kDebug ) << "Cought exception in FWMagField::checkFiledInfo\n";
   }

   if (!available)
   {
      m_source = kNone;
      fwLog( fwlog::kDebug ) << "No magnetic field info available in Event\n";
   }
}


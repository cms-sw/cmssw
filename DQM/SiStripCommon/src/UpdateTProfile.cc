#include "DQM/SiStripCommon/interface/UpdateTProfile.h"
#include "TProfile.h"
#include <iostream>
#include <math.h>


// -----------------------------------------------------------------------------
/** */
UpdateTProfile::UpdateTProfile() {;}

// -----------------------------------------------------------------------------
/** */
UpdateTProfile::~UpdateTProfile() {;}

// -----------------------------------------------------------------------------
/** */
void UpdateTProfile::setBinContents( TProfile* const prof,
				     const uint32_t& bin, 
				     const double& num_of_entries, 
				     const double& sum_of_contents,
				     const double& sum_of_squares ) {
  //   std::cout << "[UpdateTProfile::setBinContents]" << std::endl;
  
  double mean = 0.;
  double spread = 0.;
  if ( num_of_entries ) { 
    mean = sum_of_contents / num_of_entries;
    spread = sqrt( sum_of_squares/ num_of_entries - mean * mean ); 
  }
   
//   LogTrace("TEST")
//     << "[UpdateTProfile::setBinContents]"
//     << " bin: " << bin
//     << " entries: " << num_of_entries
//     << " contents: " << sum_of_contents
//     << " squared: " << sum_of_squares
//     << " mean: " << mean
//     << " spread: " << spread;
  
  UpdateTProfile::setBinContent( prof, bin, num_of_entries, mean, spread );
  
}

// -----------------------------------------------------------------------------
/** 
    Error option "s" means that the error returned using the
    GetBinError() method is the spread for that bin. (If the default
    error option is used, the error returned is spread / sqrt(N),
    where N is the number of entries for that bin). The commissioning
    procedures use option "s" throughout.

    In order that the profile histo correctly displays the "bin_mean"
    and "bin_spread" (an example being: peds +/- strip noise), we
    have to manipulate the "contents" and "error" data as follows:
    
    TProfile::SetBinEntries( bin_number, bin_entries ) 
    TProfile::SetBinContents( bin_number, bin_mean * bin_entries ) 
    TProfile::SetBinError( bin_number, weight ) 

    "weight" is calculated based on the GetBinError() method, which
    returns an error ("err" below) based on the following equation:
    
    1) if error option is set to "s":
    err = sqrt( wei^2 / num - ( sum / num )^2 ) 
    => err = sqrt( wei^2 / num - sum^2 / num^2 ) 
    => wei = sqrt( err^2 * num + sum^2 / num )
    => wei = sqrt( err^2 * num + mean^2 * num )
    
    2) else if error option is set to "" (ie, default):
    err = ( 1 / sqrt( num ) ) * sqrt( wei^2 / num - ( sum / num )^2 ) 
    => err = sqrt( wei^2 / num^2 - sum^2 / num^3 ) 
    => wei = sqrt( err^2 * num^2 + cont^2 / num )
    => wei = sqrt( err^2 * num^2 + mean^2 * num )

    where:
    "num" is the bin entries, as set by the method SetBinEntries()
    "sum" is the bin content, as set by the method SetBinContent()
    "wei" is the bin error, as set by the method SetBinError()
    and "mean" = sum / num

*/
void UpdateTProfile::setBinContent( TProfile* const prof,
				    const uint32_t& bin, 
				    const double& num_of_entries, 
				    const double& mean,
				    const double& spread ) {
  //   std::cout << "[UpdateTProfile::setBinContents]" << std::endl;

  // Check histo exists
  if ( !prof ) {
    std::cerr << "[UpdateTProfile::setBinContents]"
	 << " NULL pointer to TProfile object!" << std::endl;
    return;
  }

  // Check bin number is valid
  if ( bin == 0 || bin > static_cast<uint32_t>(prof->GetNbinsX()) ) {
    std::cerr << "[UpdateTProfile::setBinContents]"
	 << " Unexpected bin number!" << std::endl;
    return;
  }

  // Check entries are present
  if ( num_of_entries <= 0. ) { return; } //@@ what about negative weights???
  double entries = num_of_entries;
  
  // Check error option
  const char* spread_option = "s";
  const char* default_option = "";
  //   const char* option = prof->GetErrorOption();
  //   if ( option[0] != spread_option[0] ) {
  //     std::cout << "[UpdateTProfile::setBinContents]"
  // 	 << " Setting error option for TProfile to 's'!" << std::endl;
  //     prof->SetErrorOption( "s" );
  //   }
  const char* error_option = prof->GetErrorOption();

  // Calculate "weight" used for SetBinError() method
  double weight;
  if ( error_option[0] == spread_option[0] ) {
    weight = sqrt( mean*mean*entries + spread*spread*entries  );
  } else if (error_option[0] == default_option[0] ) {
    weight = sqrt( mean*mean*entries + spread*spread*entries*entries );
  } else { 
    std::cerr << "[UpdateTProfile::setBinContents]"
	 << " Unexpected error option for TProfile!" << std::endl;
    weight = 0.; 
  }
  
  // Set bin entries, contents and error
  prof->SetBinEntries( bin, entries );
  prof->SetBinContent( bin, mean * entries );
  prof->SetBinError( bin, weight );
 
//   LogTrace("TEST")
//     << "[UpdateTProfile::" << __func__ << "]"
//     << " bin/entries/mean/content/error: " 
//     << bin << "/"
//     << entries << "/"
//     << mean << "/"
//     << mean*entries << "/"
//     << weight;

  // Set total number of entries 
  if ( bin == 1 ) { prof->SetEntries( entries ); }
  else { prof->SetEntries( prof->GetEntries() + entries ); }
  
}



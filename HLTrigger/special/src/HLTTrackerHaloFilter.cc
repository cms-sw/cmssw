///////////////////////////////////////////////////////
//
// HLTTrackerHaloFilter
//
// See header file for infos on input parameters
// Comments on the code flow are in the cc file
//
// S.Viret: 01/03/2011 (viret@in2p3.fr)
//
///////////////////////////////////////////////////////

#include "HLTrigger/special/interface/HLTTrackerHaloFilter.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


//
// constructors and destructor
//
 
HLTTrackerHaloFilter::HLTTrackerHaloFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_     (config.getParameter<edm::InputTag>("inputTag")),
  max_clusTp_   (config.getParameter<int>("MaxClustersTECp")),
  max_clusTm_   (config.getParameter<int>("MaxClustersTECm")),
  sign_accu_    (config.getParameter<int>("SignalAccumulation")),
  max_clusT_    (config.getParameter<int>("MaxClustersTEC")),
  max_back_     (config.getParameter<int>("MaxAccus")),
  fastproc_     (config.getParameter<int>("FastProcessing"))
{
}

HLTTrackerHaloFilter::~HLTTrackerHaloFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTTrackerHaloFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.
  
  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);
  
  // get hold of products from Event
  edm::Handle<edm::RefGetter<SiStripCluster> > refgetter;
  event.getByLabel(inputTag_, refgetter);
  

  /// First initialize some variables

  for (int i=0;i<5;++i)
  {
    for (int j=0;j<8;++j)
    {
      SST_clus_PROJ_m[i][j]=0;
      SST_clus_PROJ_p[i][j]=0;

      for (int k=0;k<9;++k) SST_clus_MAP_m[i][j][k]=0;
      for (int k=0;k<9;++k) SST_clus_MAP_p[i][j][k]=0;
    }
  }
  
  int n_total_clus  = 0;
  int n_total_clusp = 0;
  int n_total_clusm = 0;
  int index         = 0;
  int compteur      = 0;

  int maxm          = 0;
  int maxp          = 0;

  int npeakm        = 0;
  int npeakp        = 0;

  edm::RefGetter<SiStripCluster>::const_iterator iregion = refgetter->begin();


  // Then we loop over tracker cabling regions

  for(;iregion!=refgetter->end();++iregion) 
  {

    // Don't go further if one of the TEC cluster cut is not passed
    if (n_total_clus>max_clusT_)   return false;
    if (n_total_clusp>max_clusTp_) return false;
    if (n_total_clusm>max_clusTm_) return false;

    ++index;  
    
    // Some cuts applied if fast processing requested
    //
    // !!!! Will fail if cabling is changing !!!!

    if (fastproc_ && maxm<sign_accu_ && index>1280) return false;
    if (fastproc_ && HLTTrackerHaloFilter::m_TEC_cells[compteur] != index) continue;

    ++compteur;

    // No cluster, stop here
    if ((*iregion).finish()<=(*iregion).start()) continue;

    // Look at the DetId, as we perform the quest only in TEC
    const SiStripDetId mDetId((*iregion).begin()->geographicalId());
    if (mDetId.subDetector() != SiStripDetId::TEC) continue;


    // We are in TEC, so we load the clusters
    edm::RegionIndex<SiStripCluster>::const_iterator iclus = iregion->begin();
  
    for(;iclus!=iregion->end();++iclus) 
    {
      if (iclus->geographicalId()%2 == 1) continue;

      // We skip a a part of the clusters, to restore 
      // some symmetry between rings  
      if (tTopo->tecRing(iclus->geographicalId())<3 ||tTopo->tecIsStereo(iclus->geographicalId())) continue;

      ++n_total_clus;

      int r_id = tTopo->tecRing(iclus->geographicalId())-3;
      int p_id = tTopo->tecPetalNumber(iclus->geographicalId())-1;
      int w_id = tTopo->tecWheel(iclus->geographicalId())-1;
    
      // Then we do accumulations and cuts 'on the fly'

      if (tTopo->tecSide(iclus->geographicalId())==1) // Minus side (BEAM2)
      {
	++n_total_clusm;

	if (!SST_clus_MAP_m[r_id][p_id][w_id]) // A new cell is touched
	{
	  ++SST_clus_MAP_m[r_id][p_id][w_id];	
	  ++SST_clus_PROJ_m[r_id][p_id]; // Accumulation

	  if (SST_clus_PROJ_m[r_id][p_id]>maxm) maxm = SST_clus_PROJ_m[r_id][p_id];
	  if (SST_clus_PROJ_m[r_id][p_id]==sign_accu_) ++npeakm;

	  if (npeakm>=max_back_) return false; // Too many accumulations (PKAM) 
	}
      }
      else // Plus side (BEAM1)
      {
	++n_total_clusp;

	if (!SST_clus_MAP_p[r_id][p_id][w_id])
	{
	  ++SST_clus_MAP_p[r_id][p_id][w_id];	
	  ++SST_clus_PROJ_p[r_id][p_id];

	  if (SST_clus_PROJ_p[r_id][p_id]>maxp) maxp = SST_clus_PROJ_p[r_id][p_id];
	  if (SST_clus_PROJ_p[r_id][p_id]==sign_accu_) ++npeakp;

	  if (npeakp>=max_back_) return false;  
	}
      }
    } // End of clusters loop
  } // End of regions loop
  

  // The final selection is applied here
  // Most of the cuts have already been applied tough

  if (n_total_clus>max_clusT_)                return false;
  if (n_total_clusp>max_clusTp_)              return false;
  if (n_total_clusm>max_clusTm_)              return false;
  if (n_total_clusp<sign_accu_)               return false;
  if (n_total_clusm<sign_accu_)               return false;
  if (maxm<sign_accu_ || maxp<sign_accu_)     return false;
  if (npeakm>=max_back_ || npeakp>=max_back_) return false;     

  return true;
}


// Here we define the array containing the TEC regions.

const int HLTTrackerHaloFilter::m_TEC_cells[1450] = {6,7,8,15,16,17,23,24,25,32,33,34,39,40,41,47,48,49,56,57,58,64,65,66,73,74,75,81,82,83,89,90,91,98,99,100,106,107,108,115,116,117,123,124,125,131,132,133,140,141,142,148,149,150,157,158,159,165,166,167,172,173,174,175,176,177,180,181,182,183,184,185,186,191,192,193,194,195,196,199,200,201,202,203,204,205,210,211,212,213,214,215,219,220,221,222,223,224,225,228,229,230,231,232,233,234,239,240,241,242,243,244,248,249,250,251,252,253,254,259,260,261,262,263,264,268,269,270,271,272,273,274,278,279,280,281,282,283,284,289,290,291,292,293,294,298,299,300,301,302,303,304,308,309,310,311,312,313,314,319,320,321,322,323,324,328,329,330,331,332,333,334,339,340,341,342,343,344,348,349,350,351,352,353,354,358,359,360,361,362,363,364,368,369,370,371,372,373,374,375,376,379,380,381,382,383,384,385,386,387,391,392,393,394,395,396,397,398,399,402,403,404,405,406,407,408,409,410,414,415,416,417,418,419,420,421,422,426,427,428,429,430,431,432,433,434,437,438,439,440,441,442,443,444,445,449,450,451,452,453,454,455,456,457,460,461,462,463,464,465,466,467,468,472,473,474,475,476,477,478,479,480,484,485,486,487,488,489,490,491,492,495,496,497,498,499,500,501,502,503,507,508,509,510,511,512,513,514,515,518,519,520,521,522,523,524,525,526,530,531,532,533,534,535,536,537,538,542,543,544,545,546,547,548,549,550,553,554,555,556,557,558,559,560,561,565,566,567,568,569,570,571,572,573,576,577,578,579,580,581,582,583,584,588,589,590,591,592,593,594,595,596,601,602,603,604,605,606,607,608,609,614,615,616,617,618,619,620,621,622,627,628,629,630,631,632,633,634,635,640,641,642,643,644,645,646,647,648,653,654,655,656,657,658,659,660,661,666,667,668,669,670,671,672,673,674,679,680,681,682,683,684,685,686,687,692,693,694,695,696,697,698,699,700,705,706,707,708,709,710,711,712,713,718,719,720,721,722,723,724,725,726,731,732,733,734,735,736,737,738,739,744,745,746,747,748,749,750,751,752,757,758,759,760,761,762,763,764,765,770,771,772,773,774,775,776,777,778,783,784,785,786,787,788,789,790,791,796,797,798,799,800,801,802,803,804,809,810,811,812,813,814,815,816,817,822,823,824,825,826,827,828,829,830,835,836,837,838,839,840,841,842,843,848,849,850,851,852,853,854,855,856,862,863,864,865,866,867,873,874,875,876,877,878,884,885,886,887,888,889,895,896,897,898,899,900,906,907,908,909,910,911,917,918,919,920,921,922,928,929,930,931,932,933,939,940,941,942,943,944,950,951,952,953,954,955,961,962,963,964,965,966,972,973,974,975,976,977,983,984,985,986,987,988,994,995,996,997,998,999,1005,1006,1007,1008,1009,1010,1016,1017,1018,1019,1020,1021,1027,1028,1029,1030,1031,1032,1038,1039,1040,1041,1042,1043,1049,1050,1051,1052,1053,1054,1060,1061,1062,1063,1064,1065,1071,1072,1073,1074,1075,1076,1084,1085,1086,1094,1095,1103,1104,1105,1113,1114,1122,1123,1124,1132,1133,1134,1142,1143,1144,1152,1153,1154,1162,1163,1171,1172,1173,1181,1182,1183,1191,1192,1193,1201,1202,1203,1211,1212,1220,1221,1222,1230,1231,1232,1240,1241,1249,1250,1251,1259,1260,1261,1269,1270,1271,2823,2824,2825,2833,2834,2835,2842,2843,2844,2851,2852,2859,2860,2861,2868,2869,2870,2877,2878,2879,2886,2887,2888,2896,2897,2905,2906,2907,2915,2916,2917,2925,2926,2934,2935,2936,2944,2945,2946,2954,2955,2956,2964,2965,2966,2974,2975,2983,2984,2985,2993,2994,2995,3003,3004,3005,3011,3012,3013,3014,3015,3016,3022,3023,3024,3025,3026,3027,3033,3034,3035,3036,3037,3038,3044,3045,3046,3047,3048,3049,3055,3056,3057,3058,3059,3060,3066,3067,3068,3069,3070,3071,3077,3078,3079,3080,3081,3082,3088,3089,3090,3091,3092,3093,3099,3100,3101,3102,3103,3104,3110,3111,3112,3113,3114,3115,3121,3122,3123,3124,3125,3126,3132,3133,3134,3135,3136,3137,3143,3144,3145,3146,3147,3148,3154,3155,3156,3157,3158,3159,3165,3166,3167,3168,3169,3170,3176,3177,3178,3179,3180,3181,3187,3188,3189,3190,3191,3192,3198,3199,3200,3201,3202,3203,3209,3210,3211,3212,3213,3214,3220,3221,3222,3223,3224,3225,3230,3231,3232,3233,3234,3235,3236,3237,3238,3243,3244,3245,3246,3247,3248,3249,3250,3251,3256,3257,3258,3259,3260,3261,3262,3263,3264,3269,3270,3271,3272,3273,3274,3275,3276,3277,3282,3283,3284,3285,3286,3287,3288,3289,3290,3295,3296,3297,3298,3299,3300,3301,3302,3303,3308,3309,3310,3311,3312,3313,3314,3315,3316,3321,3322,3323,3324,3325,3326,3327,3328,3329,3334,3335,3336,3337,3338,3339,3340,3341,3342,3347,3348,3349,3350,3351,3352,3353,3354,3355,3360,3361,3362,3363,3364,3365,3366,3367,3368,3373,3374,3375,3376,3377,3378,3379,3380,3381,3386,3387,3388,3389,3390,3391,3392,3393,3394,3399,3400,3401,3402,3403,3404,3405,3406,3407,3412,3413,3414,3415,3416,3417,3418,3419,3420,3425,3426,3427,3428,3429,3430,3431,3432,3433,3438,3439,3440,3441,3442,3443,3444,3445,3446,3451,3452,3453,3454,3455,3456,3457,3458,3459,3464,3465,3466,3467,3468,3469,3470,3471,3472,3477,3478,3479,3480,3481,3482,3483,3484,3485,3489,3490,3491,3492,3493,3494,3495,3496,3497,3500,3501,3502,3503,3504,3505,3506,3507,3508,3511,3512,3513,3514,3515,3516,3517,3518,3519,3522,3523,3524,3525,3526,3527,3528,3529,3530,3533,3534,3535,3536,3537,3538,3539,3540,3541,3545,3546,3547,3548,3549,3550,3551,3552,3553,3556,3557,3558,3559,3560,3561,3562,3563,3564,3568,3569,3570,3571,3572,3573,3574,3575,3576,3579,3580,3581,3582,3583,3584,3585,3586,3587,3591,3592,3593,3594,3595,3596,3597,3598,3599,3603,3604,3605,3606,3607,3608,3609,3610,3611,3614,3615,3616,3617,3618,3619,3620,3621,3622,3626,3627,3628,3629,3630,3631,3632,3633,3634,3637,3638,3639,3640,3641,3642,3643,3644,3645,3649,3650,3651,3652,3653,3654,3655,3656,3657,3661,3662,3663,3664,3665,3666,3667,3668,3669,3672,3673,3674,3675,3676,3677,3678,3679,3680,3684,3685,3686,3687,3688,3689,3690,3691,3692,3695,3696,3697,3698,3699,3700,3701,3702,3703,3707,3708,3709,3710,3711,3712,3713,3714,3715,3720,3721,3722,3723,3724,3725,3729,3730,3731,3732,3733,3734,3739,3740,3741,3742,3743,3744,3748,3749,3750,3751,3752,3753,3758,3759,3760,3761,3762,3763,3768,3769,3770,3771,3772,3773,3776,3777,3778,3779,3780,3781,3782,3787,3788,3789,3790,3791,3792,3795,3796,3797,3798,3799,3800,3801,3806,3807,3808,3809,3810,3811,3815,3816,3817,3818,3819,3820,3821,3825,3826,3827,3828,3829,3830,3831,3836,3837,3838,3839,3840,3841,3845,3846,3847,3848,3849,3850,3851,3855,3856,3857,3858,3859,3860,3861,3866,3867,3868,3869,3870,3871,3875,3876,3877,3878,3879,3880,3881,3886,3887,3888,3889,3890,3891,3895,3896,3897,3898,3899,3900,3905,3906,3907,3908,3909,3910,3915,3916,3917,3924,3925,3926,3931,3932,3933,3940,3941,3942,3947,3948,3949,3954,3955,3956,3963,3964,3965,3971,3972,3973,3980,3981,3982,3988,3989,3990,3996,3997,3998,4005,4006,4007,4013,4014,4015,4022,4023,4024,4030,4031,4032,4038,4039,4040,4047,4048,4049,4055,4056,4057,4064,4065,4066,4071,4072,4073};

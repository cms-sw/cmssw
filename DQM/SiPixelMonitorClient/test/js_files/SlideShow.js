var SlideShow = {};

//____________________________________________________________________
//____________________________________________________________________
var DEBUG            = true;
var slideListFiles   = new Array();
var slideListTitles  = new Array();
var BarrelListFiles  = new Array();
var BarrelListTitles = new Array();
var EndcapListFiles  = new Array();
var EndcapListTitles = new Array();
var ErrorListFiles   = new Array();
var ErrorListTitles  = new Array();
var slideShowSpeed   = 3000;  // miliseconds
var index            = 0;
var nSlides          = 0;
var MAX_SLIDES       = 20;
var timerID;
//____________________________________________________________________
//____________________________________________________________________

SlideShow.changeUpdateInterval = function(){
 try{

  var theRefresh  = document.getElementById("update_slideshow");
  slideShowSpeed  = theRefresh.options[theRefresh.selectedIndex].value;
  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________
SlideShow.StartSlideShow = function() {
 try{

   SlideShow.changeUpdateInterval();
   SlideShow.setSlide(index);
   index = (index+1) % nSlides;
   timerID = setTimeout('SlideShow.StartSlideShow()', slideShowSpeed);
  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________

SlideShow.StopSlideShow = function() {
  try{
    if (timerID != null){ 
      clearTimeout(timerID);
    }
  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________

SlideShow.ShowFirst = function() {
 try{
   SlideShow.setSlide(0);
  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________

SlideShow.ShowLast = function() {
 try{
   SlideShow.setSlide(nSlides-1);
  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________

SlideShow.ShowPrev = function() {
 try{
   index = (index-1) % nSlides;
   if (index<0) index = nSlides-1;
   SlideShow.setSlide(index);
  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________

SlideShow.ShowNext = function() {
 try{
   index = (index+1) % nSlides;
   SlideShow.setSlide(index);
  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________

SlideShow.setSlide = function(index) {
 try{
   if (nSlides == 0) {
     if (DEBUG) alert("Noone slide stored yet!");
     return false;
   }

//   var queryString;
//   var url = WebLib.getApplicationURL2();
//   url += "/Request?";
//   queryString = "RequestID=PlotHistogramFromLayout";
//   url += queryString;
//   WebLib.makeRequest(url, WebLib.dummy);

   $('imageCanvas').imageList     = slideListFiles;
   $('imageCanvas').titlesList    = slideListTitles;
   $('imageCanvas').current_start = index;
 
   IMGC.IMAGES_PER_ROW      = 1;
   IMGC.IMAGES_PER_COL      = 1;
   IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
   IMGC.ASPECT_RATIO        = 1.5 ;
   IMGC.THUMB_MICROFICATION = 4 ;
   IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH ;
   IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO);
   IMGC.PATH_TO_PICTURES    = "images/" ;
 
   IMGC.computeCanvasSize() ;

  }catch(e){
   alert(e.message);
  }
}

//____________________________________________________________________
SlideShow.SummaryOverView = function(type)
{
  try{

    if(type=="Pixel"){
      $('imageCanvas').imageList  = slideListFiles;
      $('imageCanvas').titlesList = slideListTitles;
    }
    if(type=="Barrel"){
      $('imageCanvas').imageList  = BarrelListFiles;
      $('imageCanvas').titlesList = BarrelListTitles;
    }
    if(type=="Endcap"){
      $('imageCanvas').imageList  = EndcapListFiles;
      $('imageCanvas').titlesList = EndcapListTitles;
    }
    if(type=="Error"){
      $('imageCanvas').imageList  = ErrorListFiles;
      $('imageCanvas').titlesList = ErrorListTitles;
    }
    $('imageCanvas').current_start = 0;
  
    SlideShow.computeImagesPerCanvas($('imageCanvas').imageList);
    IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
    IMGC.ASPECT_RATIO        = 1.5 ;
    IMGC.THUMB_MICROFICATION = 4 ;
    IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH ;
    IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO);
    IMGC.PATH_TO_PICTURES    = "images/" ;
  
    IMGC.computeCanvasSize() ;
 
    SlideShow.StopSlideShow();

  }catch(e){
   alert(e.message);
  }
}

//____________________________________________________________________
SlideShow.ErrorOverview = function()
{
  try{

    $('imageCanvas').imageList     = ErrorListFiles;
    $('imageCanvas').titlesList    = ErrorListTitles;
    $('imageCanvas').current_start = 0;
  
    SlideShow.computeImagesPerCanvas(BarrelListFiles);
    IMGC.IMAGES_PER_PAGE     = IMGC.IMAGES_PER_ROW * IMGC.IMAGES_PER_COL;
    IMGC.ASPECT_RATIO        = 1.5 ;
    IMGC.THUMB_MICROFICATION = 4 ;
    IMGC.BASE_IMAGE_WIDTH    = IMGC.DEF_IMAGE_WIDTH ;
    IMGC.BASE_IMAGE_HEIGHT   = parseInt(IMGC.BASE_IMAGE_WIDTH / IMGC.ASPECT_RATIO);
    IMGC.PATH_TO_PICTURES    = "images/" ;
  
    IMGC.computeCanvasSize() ;
 
    SlideShow.StopSlideShow();

  }catch(e){
   alert(e.message);
  }
}
//____________________________________________________________________
SlideShow.computeImagesPerCanvas = function(list)
{
  try{
    var numberOfImages = list.length;
    if(numberOfImages == 1){
      IMGC.IMAGES_PER_ROW = 1;
      IMGC.IMAGES_PER_COL = 1;
    }
    if(numberOfImages == 2){
      IMGC.IMAGES_PER_ROW = 1;
      IMGC.IMAGES_PER_COL = 2;
    }
    if(numberOfImages == 3 || numberOfImages == 4){
      IMGC.IMAGES_PER_ROW = 2;
      IMGC.IMAGES_PER_COL = 2;
    }
    if(numberOfImages == 5 || numberOfImages == 6){
      IMGC.IMAGES_PER_ROW = 2;
      IMGC.IMAGES_PER_COL = 3;
    }
    if(numberOfImages >= 7){
      IMGC.IMAGES_PER_ROW = 3;
      IMGC.IMAGES_PER_COL = 3;
    }
  }catch(e){
    alert(e.message);
  }
}

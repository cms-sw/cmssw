//____________________________________________________________________
// A set of javascript functions to display a set of selected plots
// as a contineous show of slides. A number of slide show actions include
//  - Start, Stop, FirstSlide, LastSlide, Previous, Next
//  - Set a specific Slide
//
//____________________________________________________________________
var SlideShow = {};
//____________________________________________________________________
// Static variables
//
SlideShow.DEBUG = true;
SlideShow.slideImageList  = new Array();
SlideShow.slideTitleList = new Array();
SlideShow.slideShowSpeed = 1000;  // miliseconds
SlideShow.index = 0;
SlideShow.nSlides = 0;
SlideShow.MAX_SLIDES = 20;
SlideShow.timerID = null;
//____________________________________________________________________

SlideShow.ChangeUpdateInterval = function(){
  try{

    var theRefresh  = document.getElementById("update_slideshow");
    if (theRefresh == null) {
      SlideShow.slideShowSpeed = 10000;
    } else { 
      SlideShow.slideShowSpeed = theRefresh.options[theRefresh.selectedIndex].value;
    }
  }catch(e){
    alert("[SlideShow::ChangeUpdateInterval] " + e.message);
  }
}
//____________________________________________________________________
// Start the slide show with full list of images
//
SlideShow.StartSlideShow = function()
{
  try{

    SlideShow.ChangeUpdateInterval();
    SlideShow.setSlide(SlideShow.index);
    SlideShow.index = (SlideShow.index + 1) % SlideShow.nSlides;
    SlideShow.timerID = setTimeout('SlideShow.StartSlideShow()', SlideShow.slideShowSpeed);
  }catch(e){
    alert("[SlideShow::StartSlideShow] " + e.message);
  } 
}
//____________________________________________________________________
// Stop the slide show 
//
SlideShow.StopSlideShow = function()
{
  try{

    if (SlideShow.timerID != null) {
      clearTimeout(SlideShow.timerID);
    }
  }catch(e){
   alert("[SlideShow::StopSlideShow] " + e.message);
  }
}
//____________________________________________________________________
// Show first slide
//
SlideShow.ShowFirst  = function()
{
  try{
  
   SlideShow.setSlide(0);
  }catch(e){
   alert("[SlideShow::ShowFirst] " + e.message);
  }
}
//____________________________________________________________________
// Show last slide
//
SlideShow.ShowLast = function()
{
  try{

   SlideShow.setSlide(SlideShow.nSlides-1);
  }catch(e){
   alert("[SlideShow::setSlide] " + e.message);
  }
}
//____________________________________________________________________
// Show previous slide
//
SlideShow.ShowPrev = function()
{
 try{
   SlideShow.index = (SlideShow.index-1) % nSlides;
   if (SlideShow.index < 0) SlideShow.index = SlideShow.nSlides - 1;
   SlideShow.setSlide(SlideShow.index);
  }catch(e){
   alert("[SlideShow::setSlide] " + e.message);
  }

}
//____________________________________________________________________
// Show next slide
//
SlideShow.ShowNext = function()
{
 try{
   SlideShow.index = (SlideShow.index + 1) % nSlides;
   SlideShow.setSlide(SlideShow.index);
  }catch(e){
   alert("[SlideShow::setSlide] " + e.message);
  }
}
//____________________________________________________________________
// Set a specific slide
//
SlideShow.setSlide = function(index) 
{
 try{
   if (SlideShow.nSlides == 0) {
     if (SlideShow.DEBUG) alert("[SlideShow::setSlide] " + "[SlideShow::setSlide] Empty Image List!");
     return false;
   }
  var urlImageList = SlideShow.slideImageList[SlideShow.index];
  var urlTitleList = SlideShow.slideTitleList[SlideShow.index];
  var getTitles = new Ajax.Request(urlTitleList,	   // Load titles first, because they are
  				  {			   // used by the IMGC.processImageList
  				   method: 'get',	   // which fires later on
  				   parameters: '', 
  				   onComplete: IMGC.processTitlesList // <-- call back function
  				  });
  var getFiles  = new Ajax.Request(urlImageList, 
  				  {
  				   method: 'get', 
  				   parameters: '', 
  				   onComplete: IMGC.processImageList  // <-- call back function
  				  });
  }catch(e){
   alert("[SlideShow::setSlide] " + e.message);
  }
}


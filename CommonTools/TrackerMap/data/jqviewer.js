    var layer;
    var crate;
    var remotewin;
    layer=1;crate=1; 
      var $layerp = {};
      var $layer = {};
      var $cratep = {};
      var $crate = {};
      var numl;
      var $tmap ;
      var $fedtmap ;
      var $detail,$cdetail;
      var imgwidth;
      var imgheight;
      var imgleft;
      var imgtop;
      var iconwidth;
      var xoffset=10;
      var yoffset=60;
      var iconheight;
      var iconspacing; 
    function setip(url){
        $("#ctip")
         .attr('src',url)
 ;   
    }
    function setip1(url){
        $("#ctip1")
         .attr('src',url)
 ;   
    }
    function makeRemote(){
    remote=window.open("","remote window",width=500,height=500);
    remote.location.href=servername+tmapname+"crate1.xml";
    if(remote.opener==null) remote.opener=window;
    remote.opener.name="opener";
    return remote;
    }
    remotewin=makeRemote();
$(document).ready(function(){
        function createControl(src,x,y){
         return $('<img/>')
          .attr('src',src)
          .addClass('controls')
          .css('left',x)
          .css('top',y)
          .css('position','absolute') 
          .css('display','none')
          .css('z-index','9')
          .appendTo('#controls');
        }
        function createCControl(src,x,y){
         return $('<img/>')
          .attr('src',src)
          .addClass('ccontrols')
          .css('left',x)
          .css('top',y)
          .css('position','absolute') 
          .css('display','none')
          .css('z-index','9')
          .appendTo('#ccontrols');
        }
     
   function createPlaceHolder(src,x,y,w,h,title){
           var w1=Math.round(w)+'px';
           var h1=Math.round(h)+'px';
     //      alert(x+" "+y+" "+w1+" "+h1);
           return $('<img/>')
          .attr('src',src)
          .attr('title',title)
          .addClass('place')
          .css({
             'opacity' : 0.0,    
             'left': x,
             'top' : y,
             'width': w1,
             'height': h1,
             'position':'absolute',
              'z-index':'7'
           })
         .appendTo('#place');
        }
        function createIcon(src,x,y,w,h){
           var w1=Math.round(w)+'px';
           var h1=Math.round(h)+'px';
    //      alert(x+" "+y+" "+w+" "+h);
          return $('<img/>')
          .attr('src',src)
          .addClass('icons')
          .css({
             'opacity' : 0.6, 
             'display':'none',    
             'left': x,
             'top' : y,
             'width': w1,
             'height': h1,
             'position':'absolute',
              'z-index':'6'
           })
          .appendTo('#icons');
        }
        function createCPlaceHolder(src,x,y,w,h){
         var w1=Math.round(w)+'px';
         var h1=Math.round(h)+'px';
         //  alert(x+" "+y+" "+w1+" "+h1);
         return $('<img/>')
          .attr('src',src)
          .addClass('cplace')
          .css({
             'opacity' : 0.0, 
             'left': x,
             'top' : y,
             'width': w1,
             'height': h1,
             'position':'absolute',
              'z-index':'6'
           })
          .appendTo('#cplace');
        }
        function createCIcon(src,x,y,w,h){
         var w1=Math.round(w)+'px';
         var h1=Math.round(h)+'px';
         return $('<img/>')
          .attr('src',src)
          .css({
             'opacity' : 0.6, 
             'display':'none',    
             'left': x,
             'top' : y,
             'width': w1,
             'height': h1,
             'position':'absolute',
              'z-index':'6'
           })
          .addClass('cicons')
          .appendTo('#cicons');
        }
      
     function createImgCover(first)    {
      
      
       numl=0;
       $tmap = $('#content1');
       $tmap.unbind('click mouseenter mouseleave');
       $detail = $('#detail');
       imgwidth=$tmap.width();
       imgheight=$tmap.height();
       if(first==1)imgleft=$tmap.offset().left+30; else imgleft=$tmap.offset().left;
       imgtop=$tmap.offset().top;
       //alert($tmap.width()+" "+$tmap.height()+" "+$tmap.offset().left+" "+$tmap.offset().top);
        xoffset_BAR=imgwidth/11;
        xoffset_TIDP=  imgwidth/75;
        xoffset_TECP= imgwidth/6.5;
        xoffset_TIDM  = imgwidth/6.8;
        xoffset_TECM=imgwidth -imgwidth/17;
        xoffset_BAREND=imgwidth -imgwidth/11;
        iconsize_TEC=(xoffset_TECM-xoffset_TECP)/9;
        iconsize_TID=(xoffset_TIDM-xoffset_TIDP)/3;
        yoffsetFirstRow=imgheight/6.7; 
        iconheight_BAR=(imgheight-yoffsetFirstRow)/4;
        iconwidth_BAR=(xoffset_BAREND-xoffset_BAR)/5;
   //alert(imgwidth+" "+ imgheight+" "+imgleft+" "+imgtop+" "+iconwidth+" "+iconheight+" ");
     for (var xpos=imgleft+xoffset_TECM; xpos > imgleft+xoffset_TECP+iconsize_TEC/2; xpos = xpos -iconsize_TEC){
         title='TEC -z layer|';
         $layerp[numl] = createPlaceHolder('images/endcapLayer.png',xpos-iconsize_TEC,yoffsetFirstRow+imgtop+3*iconheight_BAR,iconsize_TEC,iconheight_BAR,title);
         $layer[numl] = createIcon('images/endcapLayer.png',xpos-iconsize_TEC,yoffsetFirstRow+imgtop+3*iconheight_BAR,iconsize_TEC,iconheight_BAR);
         numl++;
         }
     for (var xpos=imgleft+xoffset_TIDM; xpos > imgleft+xoffset_TIDP+iconsize_TID/2; xpos = xpos -iconsize_TID){
         title='TID -z layer|';
         $layerp[numl] = createPlaceHolder('images/endcapLayer.png',xpos-iconsize_TID,yoffsetFirstRow+imgtop+3*iconheight_BAR,iconsize_TID,iconheight_BAR,title);
         $layer[numl] = createIcon('images/endcapLayer.png',xpos-iconsize_TID,yoffsetFirstRow+imgtop+3*iconheight_BAR,iconsize_TID,iconheight_BAR);
         numl++;
         }
       // skip pixel endcap  
     for (var nl=0; nl<3;nl++){
         title='PIXE -z layer|';
         $layerp[numl] = createPlaceHolder('images/endcapLayer.png',imgleft,imgtop,1,1,title);
         $layer[numl] = createIcon('images/endcapLayer.png',imgleft,imgtop,1,1);
         numl++;
         }
     for (var nl=0; nl<3;nl++){
         title='PIXE +z layer|';
         $layerp[numl] = createPlaceHolder('images/endcapLayer.png',imgleft,imgtop,1,1,title);
         $layer[numl] = createIcon('images/endcapLayer.png',imgleft,imgtop,1,1);
         numl++;
         }
     for (var xpos=imgleft+xoffset_TIDP; xpos < imgleft+xoffset_TIDM-iconsize_TID/2; xpos = xpos +iconsize_TID){
         title='TID +z layer|';
         $layerp[numl] = createPlaceHolder('images/endcapLayer.png',xpos,yoffsetFirstRow+imgtop,iconsize_TID,iconheight_BAR,title);
         $layer[numl] = createIcon('images/endcapLayer.png',xpos,yoffsetFirstRow+imgtop,iconsize_TID,iconheight_BAR);
         numl++;
         }
     for (var xpos=imgleft+xoffset_TECP; xpos < imgleft+xoffset_TECM-iconsize_TEC/2; xpos = xpos +iconsize_TEC){
         title='TEC +z layer|';
         $layerp[numl] = createPlaceHolder('images/endcapLayer.png',xpos,yoffsetFirstRow+imgtop,iconsize_TEC,iconheight_BAR,title);
         $layer[numl] = createIcon('images/endcapLayer.png',xpos,yoffsetFirstRow+imgtop,iconsize_TEC,iconheight_BAR);
         numl++;
         }
     for (var nl=0; nl<3;nl++){
         title='PIXB layer|';
         $layerp[numl] = createPlaceHolder('images/barrelLayer.png',imgleft,imgtop,1,1,title);
         $layer[numl] = createIcon('images/barrelLayer.png',imgleft,imgtop,1,1);
         numl++;
         }
      for (var xpos=imgleft+xoffset_BAR; xpos <imgleft+xoffset_BAREND-iconwidth_BAR/2; xpos = xpos +iconwidth_BAR){
      title='barrel layer|';
           $layerp[numl] = createPlaceHolder('images/barrelLayer.png',xpos,imgtop+yoffsetFirstRow+2*iconheight_BAR,iconwidth_BAR,iconheight_BAR,title);
           $layer[numl] = createIcon('images/barrelLayer.png',xpos,imgtop+yoffsetFirstRow+2*iconheight_BAR,iconwidth_BAR,iconheight_BAR);
           numl++;
           $layerp[numl] = createPlaceHolder('images/barrelLayer.png',xpos,imgtop+yoffsetFirstRow+iconheight_BAR,iconwidth_BAR,iconheight_BAR,title);
           $layer[numl] = createIcon('images/barrelLayer.png',xpos,imgtop+yoffsetFirstRow+iconheight_BAR,iconwidth_BAR,iconheight_BAR);
           numl++;
          } 
$('.place').click(function(e){
          var index = $('.place').index(this);
          var cornerPoint = {};
          var startPoint = $detail.offset();
          startPoint.width=$detail.width; 
          startPoint.heigth=$detail.height; 
          cornerPoint.width=$detail.width+100; 
          cornerPoint.heigth=$detail.height+100; 
          cornerPoint.top=0;
          cornerPoint.left=588;
          if(index<0||index>42)alert(index); else  
          
          $detail.attr('src',tmapname+'layer'+(index+1)+'.xml').show();
          
          $tipButton
            .css('left',cornerPoint.left-30)
            .css( 'top' , cornerPoint.top)
            .css( 'z-index',9)
           .show()
           .click(function(){
           if($("#ctip").is(':visible'))$("#ctip").hide(); else $("#ctip").show();
             });
          $reloadButton
            .css('left',cornerPoint.left-15)
            .css( 'top' , cornerPoint.top)
            .css( 'z-index',9)
           .show()
           .click(function(){
            $detail.attr('src',tmapname+'layer'+(index+1)+'.xml');
             });
          $closeButton
            .css({
              'left': cornerPoint.left,
              'top' : cornerPoint.top,
              'z-index': 9
              })
           .show()
           .click(function(){
            $detail.hide();
            $closeButton.unbind('click').hide();
            $reloadButton.unbind('click').hide();
            $tipButton.unbind('click').hide();
             });
           })     
     .hover(function(){
          var index = $('.place').index(this);
          //alert(index);
          if(index<0||index>42)alert(index); else  
          $layer[index].show();
      } ,function(){
          if(index<0||index>42)alert(index); else  
          var index = $('.place').index(this);
          $layer[index].hide();
     }); 
      }
     function createcImgCover(first)    {
      numl=0;
      $fedtmap = $('#content2');
      $fedtmap.unbind('click mouseenter mouseleave');
      $cdetail = $('#cdetail');
       imgwidth=$fedtmap.width();
       imgheight=$fedtmap.height();
       if(first==1)imgleft=$fedtmap.offset().left+30; else imgleft=$fedtmap.offset().left;
       if(first==1)imgtop=$fedtmap.offset().top-516; else imgtop=$fedtmap.offset().top;
       xoffset=imgwidth/24;
       iconwidth=(imgwidth-xoffset)/9;
       iconheight=imgheight/3.6;
       iconspacing=iconheight/15;
       iconxspacing=iconwidth/80;
       yoffset=imgheight/16; 
      // alert(imgwidth+" "+ imgheight+" "+imgleft+" "+imgtop+" "+iconwidth+" "+iconheight+" "+xoffset+" "+yoffset);
      for (var xpos=imgleft+xoffset; xpos < imgleft+imgwidth; xpos = xpos +iconwidth+iconxspacing){
           $cratep[numl] = createCPlaceHolder('images/fed.png',xpos,imgtop+yoffset,iconwidth,iconheight);
           $crate[numl] = createCIcon('images/fed.png',xpos,imgtop+yoffset+iconspacing,iconwidth,iconheight);
           numl++; if (numl==ncrates) break;
           $cratep[numl] = createCPlaceHolder('images/fed.png',xpos,imgtop+iconheight+yoffset,iconwidth,iconheight);
           $crate[numl] = createCIcon('images/fed.png',xpos,imgtop+iconheight+yoffset+iconspacing,iconwidth,iconheight);
           numl++; if (numl==ncrates) break;
           $cratep[numl] = createCPlaceHolder('images/fed.png',xpos,imgtop+2*iconheight+2*iconspacing+yoffset,iconwidth,iconheight);
           $crate[numl] = createCIcon('images/fed.png',xpos,imgtop+2*iconheight+yoffset+2*iconspacing,iconwidth,iconheight);
           numl++;
          }
     $('.cplace').click(function(e){
          var index = $('.cplace').index(this);
          var cornerPoint = {};
          var startPoint = $cdetail.offset();
          startPoint.width=$cdetail.width; 
          startPoint.heigth=$cdetail.height; 
          cornerPoint.width=$cdetail.width+100; 
          cornerPoint.heigth=$cdetail.height+100; 
          cornerPoint.top=0;
          cornerPoint.left=688;
          if(index<0||index>24)alert(index); else  
          $cdetail.attr('src',tmapname+'crate'+(index+1)+'.xml').show();
          $tipButton1
            .css('left',cornerPoint.left-30)
            .css( 'top' , cornerPoint.top)
            .css( 'z-index',9)
           .show()
           .click(function(){
           if($("#ctip1").is(':visible'))$("#ctip1").hide(); else $("#ctip1").show();
             });
          $reloadButton1
            .css('left',cornerPoint.left-15)
            .css( 'top' , cornerPoint.top)
            .css( 'z-index',9)
           .show()
           .click(function(){
            $cdetail.attr('src',tmapname+'crate'+(index+1)+'.xml');
             });
          $closeButton1
            .css({
              'left': cornerPoint.left,
              'top' : cornerPoint.top,
              'z-index': 9
              })
           .show()
           .click(function(){
            $cdetail.hide();
            $closeButton1.unbind('click').hide();
            $reloadButton1.unbind('click').hide();
            $tipButton1.unbind('click').hide();
             });
           })
         .hover(function(){
          var index = $('.cplace').index(this);
          if(index<0||index>24)alert(index); else  
          $crate[index].show();
      } ,function(){
          if(index<0||index>24)alert(index); else  
          var index = $('.cplace').index(this);
          $crate[index].hide();
     });
   
      }
      
     var $closeButton = createControl('images/close.png',0,460);
     var $reloadButton = createControl('images/reload.png',0,460);
     var $tipButton = createControl('images/tip.png',325,460);
     var $closeButton1 = createCControl('images/close.png',0,460);
     var $reloadButton1 = createCControl('images/reload.png',325,460);
     var $tipButton1 = createCControl('images/tip.png',325,460);
      createcImgCover(1);
      createImgCover(1);
           
     
     $tmap.resizable({
      stop: function(event, ui) {
         $('#place').empty();
         $('#icons').empty();
         createImgCover(0); 
         }
        });
     $fedtmap.resizable({
      stop: function(event, ui) {
         $('#cplace').empty();
         $('#cicons').empty();
         createcImgCover(0); 
         }
        });
 $('#tabs').tabs();
        $("#draggable").draggable();
        $("#draggable1").draggable();
        $("#content1").resizable({ aspectRatio: 15/8 });
        $("#content2").resizable({ aspectRatio: 15/8 });
});

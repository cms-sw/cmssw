<?php

$pageTitle = "Index";

include_once("header.php");


function printPNInfo(){

echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#PNInfo').clickover({ placement: 'right'});
   });
</script>
";

echo "     
  <a href=\"#\" id='PNInfo' rel=\"clickover\" data-content=\"
	Choose a UNIQUE publication number (it could be cadi number, for example:<br/> TOP-11-061 (preferable) 
	or any other way as TOP-11-061_1 where <b>'_1'</b> could be added in case of  multiple analyses contribute to a cadi line or AN number, it should be unique)

   \" data-original-title=\"The CMS publication number <button type='button'class='close' data-dismiss='clickover'>&times;</button>\">
	<i class='icon-question-sign'></i></a>";
}




function printPasswordInfo(){

echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#passwordInfo').clickover({ placement: 'right'});
   });
</script>
";

echo "     
  <a href=\"#\" id='passwordInfo' rel=\"clickover\" data-content=\"
	<ul>
		<li>To save <b>your</b> answers.</li>
		<li>Make sure nobody else overwrites your answers.</li>
	</ul>

   \" data-original-title=\"Why do I need password? <button type='button'class='close' data-dismiss='clickover'>&times;</button>\">
	<i class='icon-question-sign'></i></a>";
}

function printEmailInfo(){

echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#emailInfo').clickover({ placement: 'right'});
   });
</script>
";

echo "     
  <a href=\"#\" id='emailInfo' rel=\"clickover\" data-content=\"
	<ul>
		<li>To identify survey's owner.</li>
		<li>Password will be sent to this email in case you forgot it.</li>
	</ul>

   \" data-original-title=\"Why do I need to provide email? <button type='button'class='close' data-dismiss='clickover'>&times;</button>\">
	<i class='icon-question-sign'></i></a>";
}

  echo "<center><h1>Analysis questionnaire</h1></center><hr/><br/>";

?>
<script>

$(document).ready(function(){
				
	$('#form').submit(function(){

	success = true;
	if (document.getElementById('password').value == '') {		
		document.getElementById('password').style.border='2px solid red';
		success = false; 
	}
	else{
		document.getElementById('password').style.border='2px solid green';
	} 

	if (document.getElementById('email').value == '' || !isEmail(document.getElementById('email').value)) {		
		document.getElementById('email').style.border='2px solid red';
		success = false; 
	}
	else{
		document.getElementById('email').style.border='2px solid green';
	}

	if (document.getElementById('publication_number').value == '') {		
		document.getElementById('publication_number').style.border='2px solid red';
		success = false; 
	}
	else{
		document.getElementById('publication_number').style.border='2px solid green';
	} 
		return success; 
	
    });
});

function isEmail(email) {
    var p = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/;
    return p.test(email);
};

</script>

<style>
.popover{
	width:250px;
}

.modal{
	width:600px;
	height:auto;
}
.modal-body{
	max-height:500px;
}
</style>
<div class="rounded" style='margin:0px auto; width:450px; margin-top:-30px; padding:5px'>
<a href="#slidesIntro" role="button" class="btn btn-primary" data-toggle="modal">Please read the introductory slides before taking this survey</a>
</div>
 
<!-- Modal -->
<div id="slidesIntro" class="modal hide fade" tabindex="-1" role="dialog" aria-labelledby="myModalLabel" aria-hidden="true">
  
  <div class="modal-body">
<div id="myCarousel" class="carousel slide">
  <!-- Carousel items -->
  <div class="carousel-inner">
    <div class="active item"><img src="slides/page_1.jpg"/></div>
    <div class="item"><img src="slides/page_2.jpg"/></div>
    <div class="item"><img src="slides/page_3.jpg"/></div>
    <div class="item"><img src="slides/page_4.jpg"/></div>
    <div class="item"><img src="slides/page_5.jpg"/></div>
  </div>
  <!-- Carousel nav -->
  <a class="carousel-control left" href="#myCarousel" data-slide="prev">&lsaquo;</a>
  <a class="carousel-control right" href="#myCarousel" data-slide="next">&rsaquo;</a>
</div>  </div>
  <div class="modal-footer">
    <button class="btn" data-dismiss="modal" aria-hidden="true">Close</button>
  </div>
</div>



<div class="rounded" style='margin:0px auto; width:970px; '>

<div class="rounded newSurvey">
	<h3>Start new survey</h3><hr style="border-color:#5A7FAD"/>

	<form id="form" action="questionnaire.php" method="POST">
		Publication number <?php  printPNInfo(); ?><br/>
		<input type="text" name="publication_number" placeholder="CMS-XYZ-YY-nnn" id="publication_number"><br/>
		Email <?php  printEmailInfo(); ?><br/>
		<input type="text" name="email" id="email"><br/>
		Password <?php  printPasswordInfo(); ?> <br/>
		<input type="text" name="password"  id="password"><br/><br/>
		<input type='hidden' name="action" value="startNew">
		<input type="submit" value="Start new survey" class="btn btn-success">
	</form>
</div>

<div class="rounded" style='margin:10px; width:623px; border:1px solid black; text-align:center; background:#7A96B9;float:left'>
	<h3>Update survey</h3>

<?php

try
  {
    //open the database
    $db = new PDO('sqlite:aq.db');
    
    $questionnaires = $db->query('SELECT * FROM Questionnaire')->fetchAll();

	echo "<center><div style='width:623px'>
	";
	echo "<div style='border-bottom:1px solid black; border-top:1px solid black; padding-top:10px; padding-bottom:10px; background:#7A96B9'>
		<div class='qcol border_right'>Publication number</div> 
		<div class='qcol border_right'>Contact</div> 
		<div class='qcol100'>Action</div>
		<div style='clear:both'></div>
	      </div>
	<div style='overflow:scroll; overflow-x: hidden; overflow-y: auto; height:216px'>
	";

    foreach($questionnaires as $questionnaire)
    {
    	
	echo "<div class='qrow'>
		<div class='qcol border_right'>".$questionnaire['publication_number']."</div> 
		<div class='qcol border_right'>".str_replace("@", "<i class='icon-globe'></i>",$questionnaire['email'])."</div>
		<div class='qcol100'>			
			<a target='_blank' href='questionnaire.php?id=".$questionnaire['id']."' class='btn btn-mini btn-success'><i class='icon-wrench icon-white'></i> Update</a>
		</div>
		<div style='clear:both'></div>
	      </div>";
    }
	
	echo "
	</div>
	</div><br/>


</center>";	
    // close the database connection
    $db = NULL;
  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }
?>

</div>
<div style="clear:both"></div>

</div>


<center>
 <hr>

<script type="text/javascript" charset="utf-8">
// Javascript
$(function() {

  var currentDate = new Date();
  $('div#clock').countdown(new Date(2013, 3, 30), function(event) {
    $this = $(this);
    switch(event.type) {
      case "seconds":
      case "minutes":
      case "hours":
      case "days":
      case "weeks":
      case "daysLeft":
        $this.find('span#'+event.type).html(event.value);
        break;
      case "finished":
        $this.fadeTo('slow', .5);
        break;
    }
  });
});
</script>

		<h2>Time left to complete your survey</h2>
	<div id="clock">
	  <p class="rounded">
	    <span id="weeks"></span>
	    Weeks
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="daysLeft"></span>
	    Days
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="hours"></span>
	    Hours
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="minutes"></span>
	    Minutes
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="seconds"></span>
	    Seconds
	  </p>
	</div>
	<h3>2013 April 30</h3>

</center>


<?php
include_once("footer.php");
?>


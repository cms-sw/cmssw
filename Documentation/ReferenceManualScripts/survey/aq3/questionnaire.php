<?php
	$pageTitle = "Questionnaire";

	include_once("header.php");


?>
<style>
.popover-title{
	display:none;
}

.popover {
	max-width:700px;
	width: auto;
	height: auto;
	max-height:200px;
}
</style>


<script>

$(document).ready(function(){
	
	
        var success = false;
			
	$('#form').submit(function(){

	var password = document.getElementById('password').value;

	if (password == '') {
		document.getElementById('password').style.border='2px solid red';
		alert('Enter password'); 
	}
	else{
        
		var qid = document.getElementById('questionnaire_id').value;
		
		
		

	$.ajax({
  		type: 'POST',
 		url: 'remind.php',
  		data: { questionnaire_id: qid, pass: password, action: 'checkpass' },
  		success: function(data) {

		        	if (data == 'TRUE'){
					success = true;
				}
				else{
					document.getElementById('password').style.border='2px solid red';
					alert('Wrong password');					
					success = false;
				}
			},
  		async:false
	    });

	
	}

        return success; 
	
    });
});


function remind(qid){
 document.getElementById('reminder').innerHTML = "Sending password <img src='img/loader.gif'/><br/>Please wait";
 document.getElementById('reminderSP').style.display = 'none';

 $.post("remind.php", { questionnaire_id: qid, action:'remindPass' },
   function(data) {
	document.getElementById('reminder').innerHTML = data;
   });

}
</script>

<?php

try
  {


    //open the database
  $db = new PDO('sqlite:aq.db');

function questionInfo($question){

if ($question['info'] != ""){

	return "

	<script type=\"text/javascript\">
	    $(window).load(function(){
	    $('#questionInfo".$question['id']."').clickover({ placement: 'top'});
	   });
	</script>

	  <a href=\"#\" id='questionInfo".$question['id']."' rel=\"clickover\" data-content=\"
		".printSafe($question['info'])."
	
	   \" class='btn btn-mini btn-primary'>
		<i class='icon-info-sign icon-white'></i> info</a>";
	}
else{
	return "";
}
}



function printOption($option, $response){


echo "<div class='optionQ'>";

	if ($option['is_checkbox'] == 1){	
		echo "<label class='checkbox'><input type='checkbox' name='checkbox".$option['id']."'";
		if ($response['selected'] == 1){echo " checked ";}
		echo " value='".$option['id']."'>".printSafe($option['title'])."</label>";
	}
	elseif ($option['is_radio'] == 1){
		echo "<label class='radio'><input type='radio' name='radio".$option['question_id']."'"; 
		if ($response['selected'] == 1){echo " checked ";}
		echo " value='".$option['id']."'>".printSafe($option['title'])."</label>";
	}	
	elseif ($option['is_text']==1) {
		echo "<label for='".$option['id']."' >".printSafe($option['title'])."</label>";
		echo "<textarea rows='3' style='width:100%' id='".$option['id']."' name='text".$option['id']."'>".printSafe($response['text'])."</textarea><br/>";
	}
	elseif ($option['is_comment']==1){
		echo printSafe($option['title'])."<br/>";
	}
	
	if ( ( ($option['is_radio'] == 1) || ($option['is_checkbox'] == 1) ) && ($option['is_commentable']==1) ){
		echo "<textarea rows='3' style='width:100%' id='".$option['id']."' name='text".$option['id']."' placeholder='Note: text will not be saved if radio/checkbox above is not selected'>".printSafe($response['text'])."</textarea><br/>";
	}
	
	
echo "</div>";
}



function printForgot($questionnaire){

	echo "
<script type=\"text/javascript\">
    $(window).load(function(){

    $('#forgot').clickover({ placement: 'top'});
   });
</script>
";


echo "     
  <a href=\"#\" id=\"forgot\" rel=\"clickover\" style='font-size: 12px;' data-content=\"	
<center>
	<span id='reminder'>Email: ".$questionnaire['email']."</span><br/>
	<button id='reminderSP' onclick='remind(".$questionnaire['id'].")'>Send password</button>
</center>
   \" data-original-title=\"<center>Password reminder</center>\">
	forgot password?
  </a>";

 		
}



function trim_text($input, $length, $ellipses = true, $strip_html = true) {
    //strip tags, if desired
    if ($strip_html) {
        $input = strip_tags($input);
    }
  
    //no need to trim, already shorter than trim length
    if (strlen($input) <= $length) {
        return $input;
    }
  
    //find last space within length
    $last_space = strrpos(substr($input, 0, $length), ' ');
    $trimmed_text = substr($input, 0, $last_space);
  
    //add ellipses (...)
    if ($ellipses) {
        $trimmed_text .= '...';
    }
  
    return $trimmed_text;
}

function printSafe($str){
	return stripslashes($str);
}

function safe($str){

	$str = str_replace('"', "&#34", $str);
	$str = str_replace("'", "&#39", $str);
	$str = str_replace("<", "&lt;", $str);
	$str = str_replace(">", "&gt;", $str);

	return $str;
}


////////////////////////

  

switch($_POST['action']){

case "startNew":

        $email = safe($_POST['email']);
        
	$password = safe($_POST['password']);

	$publication_number = safe($_POST['publication_number']);

	$questionnaire = $db->query("SELECT * FROM Questionnaire WHERE publication_number='".$publication_number."' AND email='".$email."'")->fetch();

	if ($questionnaire == null){
	
		if ( ($email != null) && ($password != null) && ($publication_number != null) ){
		
			$db->query("INSERT INTO Questionnaire (id,publication_number,email,password) VALUES (NULL,'".$publication_number."','".$email."','".$password."')");

			$questionnaire = $db->query("SELECT * FROM Questionnaire WHERE publication_number='".$publication_number."' AND email='".$email."'")->fetch();

                        echo "<script type='text/javascript'>window.location = 'questionnaire.php?id=".$questionnaire['id']."'</script>";		
		}
		else{
			echo "<script type='text/javascript'>window.location = 'index.php'</script>";		
		}
			
	}
	else{
		echo "<script type='text/javascript'>window.location = 'questionnaire.php?id=".$questionnaire['id']."'</script>";		
	}       	
        	
break;

case "save":

	$questionnaire_id = safe($_POST['questionnaire_id']);
	
	$questionnaire = $db->query("SELECT * FROM Questionnaire WHERE id='".$questionnaire_id."'")->fetch();

	
	$password = safe($_POST['password']);

	if ($questionnaire != null){

		if ($questionnaire['password'] == $password){

			$questions = $db->query('SELECT * FROM Question ORDER BY place ASC')->fetchAll();
		
			foreach($questions as $question){
        
				$options = $db->query('SELECT * FROM Option WHERE is_comment = 0 AND question_id='.$question['id'].' ORDER BY place ASC')->fetchAll();	

				foreach($options as $option){			

					$response = $db->query("SELECT * FROM Response WHERE option_id='".$option['id']."' AND questionnaire_id='".$questionnaire['id']."'")->fetch();
        			
					$selected = 0;
					$text = "";

					if ( ( ($option['is_radio'] == 1) && ($_POST['radio'.$question['id']] == $option['id']))){
						$selected = 1;
					}
					if ( ( ($option['is_checkbox'] == 1) && ($_POST['checkbox'.$option['id']] == $option['id']))){
						$selected = 1;
					}
					if ( ($option['is_text'] == 1) || ( ($option['is_commentable']==1) && ($selected==1) ) ){
						$text = safe($_POST['text'.$option['id']]);
					}
                        		
					if ($response == null){				
						$db->query("INSERT INTO Response (id,questionnaire_id,selected,text,option_id) VALUES (NULL,'".$questionnaire['id']."','".$selected."','".$text."','".$option['id']."')");
					}			
					else{				
						$db->query("UPDATE Response SET selected='".$selected."', text='".$text."' WHERE questionnaire_id='".$questionnaire['id']."' AND option_id='".$option['id']."'");
					}
				}	
			}
		}

	echo "<script type='text/javascript'>window.location = 'questionnaire.php?id=".$questionnaire_id."'</script>";

	}
	else{
		echo "<script type='text/javascript'>window.location = 'index.php'</script>";
	}
	
	
	break;

	default:
/////////////////////////
	
	echo "<div style='position:fixed; bottom:100px; right:30px'><a href='#'><u>Go to top</u></a></div>";

	$questionnaire_id = safe($_GET['id']);	

	$questionnaire = $db->query("SELECT * FROM Questionnaire WHERE id='".$questionnaire_id."'")->fetch();

	echo "<center><h1><a href='index.php'>Analysis questionnaire</a></h1></center>";

	if ( $questionnaire == null){
		echo "<hr/><center><h3>There is no such questionnaire<br/>If you think that life isn't fare and this questionnaire must be, please contact to mantas.stankevicius@cern.ch</h3></center>";
	}
	else{
  		echo "<center><h3>Publication: ".$questionnaire['publication_number']."</h3></center><hr/>";
  		
	
	

	        echo "<form id='form' action='?' method='POST'>";
		echo "<div class='container-fluid'>";
		echo "  <div class='row-fluid'>";
// Left side

		echo "    <div class='span3'>";
  		echo "       <div id='navbar'>";
  		echo "		<center><h3>Blocks</h3><hr/></center>";
  		echo "          <ul class='nav'>";
	
		$separators = $db->query('SELECT * FROM Question WHERE is_separator=1 ORDER BY place ASC')->fetchAll();  	
		$index = 1;
		foreach($separators as $separator){   
			echo "<li class='rounded separator_left'><a href='#question".$separator['id']."' title='".printSafe($separator['title'])."'><b>Block ".$index.".</b> ".printSafe(trim_text($separator['title'], 42))."</a></li>";                
			$index++;
		}
		echo "          </ul>";
		echo "       </div>";//navbar


  	echo "<div style='text-align:center;'>       	
		<hr/>";
	echo "<h4>Note!<br/> You can come any time and edit answers.</h4><br/>";

	echo"</div>";

		echo "    </div>";//span3


// Right side 	        
		echo "<div class='span8'>
		
		<input type='hidden' id='questionnaire_id' name='questionnaire_id' value='".$questionnaire['id']."'>
		<input type='hidden' name='action' value='save'>
		";  

		$questions = $db->query('SELECT * FROM Question ORDER BY place ASC')->fetchAll();
		$question_index=1;
		$block_index=1;
		foreach($questions as $question){



		$question_index_str = "";
		$separator_style = " separator_right";	
		if ($question['is_separator']==0){
			$question_index_str = "Question ".$question_index.". ";
			$separator_style = "";	
			$question_index++;
		}
		else{
			$question_index_str = "Block ".$block_index.". ";
			$block_index++;
		}

			echo "<div class='question rounded' id='question".$question['id']."'>";
			echo "<div class='rounded questionHead ".$separator_style."'>";
			echo "<div class='title'><a name='".$question['place']."'></a>".questionInfo($question)." <b>".$question_index_str."</b>".printSafe($question['title'])." </div>";
			echo "<div style='clear:both'></div>";
			echo "</div>";

			$options = $db->query('SELECT * FROM Option WHERE question_id='.$question['id'].' ORDER BY place ASC')->fetchAll();	

			foreach($options as $option){			

				$response = $db->query("SELECT * FROM Response WHERE option_id='".$option['id']."' AND questionnaire_id='".$questionnaire['id']."'")->fetch();

				printOption($option, $response);				
                        
				echo "<div style='clear:both'></div>";
	        	}

			echo "</div>"; // question


    		}
		echo "<div style='text-align:center;'>
			Password<br/>
			<input type='text' id='password' name='password' style='width:185px;' placeholder='Password'><br/>
			<button type='submit' style='height:50px; width:200px' class='btn btn-success' data-loading-text='Saving...'>Save</button><br/>";
        	printForgot($questionnaire);
		echo "</div>";

		
		echo "</div>"; // span10
		echo "</div>"; // row-fluid
		echo "</div>"; // container-fluid
	
	echo"</div>";
	echo "</form>";



	}
}//switch
	$db = NULL;

}//try
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }

        include_once("footer.php");

?>
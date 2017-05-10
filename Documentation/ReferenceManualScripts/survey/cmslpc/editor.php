
<?php

$pageTitle = "Editor";


function questionInfo($question){

if ($question['info'] != ""){

return "

<script type=\"text/javascript\">
    $(window).load(function(){
    $('#questionInfo".$question['id']."').clickover({ placement: 'top'});
   });
</script>

  <a href=\"#\" id='questionInfo".$question['id']."' rel=\"clickover\" data-content=\"
	".$question['info']."

   \" data-original-title=\"To edit press <i class='icon-wrench'></i> button on left\" class='btn btn-mini btn-primary'>
	<i class='icon-info-sign icon-white'></i> info</a>";
}

else{
	return "";
}
}


function printOption($option){
echo "<div class='optionE'>";
if ($option['is_checkbox'] == 1){	
	echo "<label class='checkbox'><input type='checkbox'>".printSafe($option['title'])."</label>";
	}
	elseif ($option['is_radio'] == 1){
		echo "<label class='radio'><input type='radio' name='radioGroup".$option['question_id']."'>".printSafe($option['title'])."</label>";
	}	
	elseif ($option['is_text']==1) {
		echo "<label for='".$option['id']."' >".printSafe($option['title'])."</label>";
		echo "<textarea rows='3' cols='50' id='".$option['id']."'></textarea><br/>";
	}
	elseif ($option['is_comment']==1){
		echo printSafe($option['title'])."<br/>";
	}
	
	if ( ( ($option['is_radio'] == 1) || ($option['is_checkbox'] == 1) ) && ($option['is_commentable']==1) ){
		echo "<textarea rows='3' cols='50' id='".$option['id']."'></textarea><br/>";
	}
	
	
echo "</div>";
}


function printAddQuestion(){

	echo "
<script type=\"text/javascript\">
    $(window).load(function(){

	$('#addQuestion').clickover({ placement: 'right'});
	$('#addSeparator').clickover({ placement: 'right'});

   });
</script>
";


echo "<div class='addQuestion'>";
echo "     
  <a href=\"#\" id=\"addQuestion\" rel=\"clickover\" class=\"btn btn-success btn-mini\" data-content=\"
	<form action='?' method='POST'>
		Title: <textarea name='title'></textarea>
		<input type='hidden' name='action' value='addQuestion'>		
		<input type='hidden' name='is_separator' value='0'>		
		<button type='submit' class='btn btn-success btn-small'>Add question</button>
		<button class='btn btn-info btn-small' data-dismiss='clickover'>Cancel</button>
	</form>
   \" data-original-title=\"New question\">
	<i class='icon-plus'></i> Add question
  </a>";

  echo "</div>";

echo "<div class='addQuestion'>";
echo "     
  <a href=\"#\" id=\"addSeparator\" rel=\"clickover\" class=\"btn btn-info btn-mini\" data-content=\"
	<form action='?' method='POST'>
		Title: <textarea name='title'></textarea>
		<input type='hidden' name='action' value='addQuestion'>
		<input type='hidden' name='is_separator' value='1'>				
		<button type='submit' class='btn btn-success btn-small'>Add separator</button>
		<button class='btn btn-info btn-small' data-dismiss='clickover'>Cancel</button>
	</form>
   \" data-original-title=\"New separator\">
	<i class='icon-plus'></i> Add separator
  </a>";

  echo "</div>";


  echo "<div style='clear:both'></div>";
}

function printAddOption($question){

	echo "
<script type=\"text/javascript\">
    $(window).load(function(){

    $('#".$question['id']."addOption').clickover({ placement: 'right'});
   });
</script>
";


echo "<div class='option'>";
echo "     
  <a href=\"#\" id=\"".$question['id']."addOption\" rel=\"clickover\" class=\"btn btn-success btn-mini\" data-content=\"
	<form action='?' method='POST'>
		Title: <textarea name='title'></textarea>
		Type:<br/>
		<label class='radio'><input type='radio' name='optionType' value='is_radio' checked> radio </label>
		<label class='radio'><input type='radio' name='optionType' value='is_checkbox'> checkbox </label>

		<label class='radio'><input type='checkbox' value='1' name='is_commentable'> commentable</label>

		<label class='radio'><input type='radio' name='optionType' value='is_text'> text field</label>

		<label class='radio'><input type='radio' name='optionType' value='is_comment'> comment (<b>not input field</b>) </label>

		<input type='hidden' name='action' value='addOption'>
		<input type='hidden' name='question_id' value='".$question['id']."'>
		<button type='submit' class='btn btn-success btn-small'>Add option</button>
		<button class='btn btn-info btn-small' data-dismiss='clickover'>Cancel</button>
	</form>
   \" data-original-title=\"New option\">
	<i class='icon-plus'></i> Add option
  </a>";




	echo "</div><div style='clear:both'></div>";
}

function printEditQuestion($question){

echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#".$question['id']."questionDelete').clickover({ placement: 'bottom'});
    $('#".$question['id']."questionEdit').clickover({ placement: 'bottom'});
   });
</script>
";

echo "<div class='edit'>";


echo "     
  <a href=\"#\" id=\"".$question['id']."questionDelete\" rel=\"clickover\" class=\"btn btn-danger btn-mini\" data-content=\"
	<form action='?' method='POST'>
		<input type='hidden' name='id' value='".$question['id']."'>
		<input type='hidden' name='action' value='deleteQuestion'>
		<button type='submit' class='btn btn-danger btn-small'>Yes, delete</button>
		<button class='btn btn-success btn-small' data-dismiss='clickover'>No, cancel</button>
	</form>
   \" data-original-title=\"Are you sure you want to delete this question?\">
	<i class='icon-trash'></i>
</a>";



echo "     
  <a href=\"#\" id=\"".$question['id']."questionEdit\" rel=\"clickover\" class=\"btn btn-info btn-mini\" data-content=\"
	<form action='?' method='POST'>
		Title<br/>
		<textarea name='title'>".printSafe($question['title'])."</textarea><br/>
		Info <br/>
		(will <b>not</b> appear if left empty)<br/>
		<textarea name='info'>".printSafe($question['info'])."</textarea>
		<input type='hidden' name='id' value='".$question['id']."'>
		<input type='hidden' name='action' value='editQuestion'><br/>
		<button type='submit' class='btn btn-success btn-small'>Submit</button>
		<button class='btn btn-small btn-info' data-dismiss='clickover'>Cancel</button>
	</form>
   \" data-original-title=\"Edit Question\">
	<i class='icon-wrench'></i>
</a>";


echo "<form action='?' method='POST' style='display:inline'>
	<input type='hidden' name='action' value='updownQuestion'>
	<input type='hidden' name='id' value='".$question['id']."'>
	<input type='hidden' name='updown' value='up'>
	<button type='submit' class='btn btn-mini btn-warning'><i class='icon-arrow-up'></i></button>
      </form>";


echo "<form action='?' method='POST' style='display:inline'>
	<input type='hidden' name='action' value='updownQuestion'>
	<input type='hidden' name='id' value='".$question['id']."'>
	<input type='hidden' name='updown' value='down'>
	<button type='submit' class='btn btn-mini btn-warning'><i class='icon-arrow-down'></i></button>
      </form>";


echo "</div>";
	
}

function printEditOption($option){

	
echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#".$option['id']."optionDelete').clickover({ placement: 'bottom'});
    $('#".$option['id']."optionEdit').clickover({ placement: 'right'});
   });
</script>
";

echo "<div class='edit'>";


echo "     
  <a href=\"#\" id=\"".$option['id']."optionDelete\" rel=\"clickover\" class=\"btn btn-danger btn-mini\" data-content=\"
	<form action='?' method='POST'>
		<input type='hidden' name='id' value='".$option['id']."'>
		<input type='hidden' name='action' value='deleteOption'>
		<button type='submit' class='btn btn-danger btn-small'>Yes, delete</button>
		<button class='btn btn-success btn-small' data-dismiss='clickover'>No, cancel</button>
	</form>
   \" data-original-title=\"Are you sure you want to delete this option?\">
	<i class='icon-trash'></i>
</a>";



echo "     
  <a href=\"#\" id=\"".$option['id']."optionEdit\" rel=\"clickover\" class=\"btn btn-info btn-mini\" data-content=\"
	<form action='?' method='POST'>
		Title<br/>
		<textarea name='title'>".printSafe($option['title'])."</textarea><br/><br/>
		Option type:<br/>
		<label class='radio'><input type='radio' name='optionType' value='is_radio'"; if($option['is_radio']){ echo " checked ";} echo"> radio </label>
		<label class='radio'><input type='radio' name='optionType' value='is_checkbox' value='1'"; if($option['is_checkbox']){ echo " checked ";} echo "> checkbox </label>
		<label class='radio'><input type='checkbox' name='is_commentable' value='1'"; if($option['is_commentable']){ echo " checked ";} echo" > commentable</label>
		<label class='radio'><input type='radio' name='optionType' value='is_text'"; if($option['is_text']){ echo " checked ";} echo "> text field</label>
		<label class='radio'><input type='radio' name='optionType' value='is_comment'"; if($option['is_comment']){ echo " checked ";} echo "> comment (<b>not input field</b>)</label>
		<input type='hidden' name='id' value='".$option['id']."'>
		<input type='hidden' name='action' value='editOption'>
		<button type='submit' class='btn btn-success btn-small'>Submit</button>
		<button class='btn btn-small btn-info' data-dismiss='clickover'>Cancel</button>
	</form>
   \" data-original-title=\"Edit option\">
	<i class='icon-wrench'></i>
</a>";


echo "<form action='?' method='POST' style='display:inline'>
	<input type='hidden' name='action' value='updownOption'>
	<input type='hidden' name='id' value='".$option['id']."'>
	<input type='hidden' name='updown' value='up'>
	<button type='submit' class='btn btn-mini btn-warning'><i class='icon-arrow-up'></i></button>
      </form>";


echo "<form action='?' method='POST' style='display:inline'>
	<input type='hidden' name='action' value='updownOption'>
	<input type='hidden' name='id' value='".$option['id']."'>
	<input type='hidden' name='updown' value='down'>
	<button type='submit' class='btn btn-mini btn-warning'><i class='icon-arrow-down'></i></button>
      </form>";


echo "</div>";
}


function returnBack($question_id){
	echo "<script type='text/javascript'>window.location = 'editor.php#question".$question_id."'</script>";
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
//    $trimmed_text = substr($input, 0, $last_space);
    $trimmed_text = substr($input, 0, $length);
  
    //add ellipses (...)
    if ($ellipses) {
        $trimmed_text .= '...';
    }
  
    return $trimmed_text;
}


function safe($str){

$str = str_replace('"', "&#34", $str);
$str = str_replace("'", "&#39", $str);
$str = str_replace("<", "&lt;", $str);
$str = str_replace(">", "&gt;", $str);
$str = str_replace("\n", "", $str);
$str = str_replace("\r", "", $str);


return $str;
}

function printSafe($str){


return stripslashes($str);
}


//////////////////////////////////////////////////////////////////////////////////////////////////

  try
  {
    //open the database
    $db = new PDO('sqlite:aq.db');
    
    


include_once("header.php");



$action = $_POST['action'];


switch($action){


	case "addQuestion":

		$title = safe($_POST['title']);
		$is_separator = safe($_POST['is_separator']);

		$count = $db->query("SELECT COUNT(*) FROM Question")->fetchColumn();
	
		$db->query("INSERT INTO Question (id,title,place,is_separator) VALUES (NULL,'".$title."','".($count+1)."','".$is_separator."')");

		$question_id = $db->query("SELECT id FROM Question WHERE place='".($count+1)."'")->fetchColumn();

		returnBack($question_id);

	break;

	case "editQuestion":
		$title = safe($_POST['title']);
		$info = safe($_POST['info']);
		$id = safe($_POST['id']);
	
		$db->query("UPDATE Question SET title='".$title."', info='".$info."' WHERE id ='".$id."'");
	
		returnBack($id);

	break;

	case "deleteQuestion":

		$id = safe($_POST['id']);

		$question = $db->query("SELECT * FROM Question WHERE id=".$id)->fetch();

		$questions = $db->query("SELECT * FROM Question WHERE place>".$question['place']."")->fetchAll();		

		foreach($questions as $q){
			$db->query("UPDATE Question SET place = '".($q['place']-1)."' WHERE id = '".$q['id']."'");
		}

		$db->query("DELETE FROM Question WHERE id='".$id."'");

		$db->query("DELETE FROM Option WHERE question_id='".$id."'");


		returnBack(0);

	break;

	case "updownQuestion":

		$id = safe($_POST['id']);

		$updown = $_POST['updown'];

		$question = $db->query("SELECT * FROM Question WHERE id=".$id)->fetch();

		$place = $question['place'];
	
		$count = $db->query("SELECT COUNT(*) FROM Question")->fetchColumn();

		if ( ($updown=="up") && ($place > 1) ){

			$question_to_push_down = $db->query("SELECT * FROM Question WHERE place='".($place-1)."'")->fetch();
			
			$db->query("UPDATE Question SET place = ".($question_to_push_down['place']+1)." WHERE id='".$question_to_push_down['id']."'");
			$db->query("UPDATE Question SET place = ".($question['place']-1)." WHERE id='".$question['id']."'");

		}
		elseif ( ($updown=="down") && ($place < $count) ){

			$question_to_push_up = $db->query("SELECT * FROM Question WHERE place='".($place+1)."'")->fetch();

			$db->query("UPDATE Question SET place = ".($question_to_push_up['place']-1)." WHERE id='".$question_to_push_up['id']."'");
			$db->query("UPDATE Question SET place = ".($question['place']+1)." WHERE id='".$question['id']."'");	

		}
	
	
		returnBack($id);
	
	break;

	case "addOption":

		$question_id = safe($_POST['question_id']);

		$title = safe($_POST['title']);

		$optionType = $_POST['optionType'];

		$is_commentable = $_POST['is_commentable'];

		$count = $db->query("SELECT COUNT(*) FROM Option WHERE question_id=".$question_id)->fetchColumn();
		

		$query = "INSERT INTO Option (id,question_id,place,title,is_text,is_radio,is_checkbox,is_commentable,is_comment) VALUES (NULL,'".$question_id."','".($count+1)."','".$title."','";
	

		if ($optionType=="is_text"){$query.="1";}else{$query.="0";}
		$query.="','";
		if ($optionType=="is_radio"){$query.="1";}else{$query.="0";}
		$query.="','";
		if ($optionType=="is_checkbox"){$query.="1";}else{$query.="0";}
		$query.="','";
		if ($is_commentable=="1"){$query.="1";}else{$query.="0";}
		$query.="','";
		if ($optionType=="is_comment"){$query.="1";}else{$query.="0";}
		$query.="')";

		
		$db->query($query);

		returnBack($question_id);
	break;

	case "editOption":

		$id = safe($_POST['id']);

		$title = safe($_POST['title']);

		$optionType = $_POST['optionType'];
		$is_commentable = $_POST['is_commentable'];		

		$question_id = $db->query("SELECT question_id FROM Option WHERE id='".$id."'")->fetchColumn();

		$query = "UPDATE Option SET title='".$title."'";
		
		$query.=", is_text='";
		if ($optionType=="is_text"){$query.="1";}else{$query.="0";}
		$query.="', is_radio='";
		if ($optionType=="is_radio"){$query.="1";}else{$query.="0";}
		$query.="', is_checkbox='";
		if ($optionType=="is_checkbox"){$query.="1";}else{$query.="0";}
		$query.="', is_commentable='";
		if ($is_commentable=="1"){$query.="1";}else{$query.="0";}
		$query.="', is_comment='";
		if ($optionType=="is_comment"){$query.="1";}else{$query.="0";}
		$query.="' WHERE id = ".$id;

		$db->query($query);
		
		returnBack($question_id);
		

	break;

	case "deleteOption":
		
		$id = safe($_POST['id']);
		
		$question_id = $db->query("SELECT question_id FROM Option where id='".$id."'")->fetchColumn();

		$option = $db->query("SELECT * FROM option WHERE id=".$id)->fetch();

		$options = $db->query("SELECT * FROM Option WHERE question_id = ".$question_id." AND place>'".$option['place']."'")->fetchAll();		

		foreach($options as $o){
			$db->query("UPDATE Option SET place = '".($o['place']-1)."' WHERE id = '".$o['id']."'");
		}

		$db->query("DELETE FROM Option WHERE id =".$id);

		returnBack($question_id);
	break;

	case "updownOption":

		$id = safe($_POST['id']);

		$updown = $_POST['updown'];

		$option = $db->query("SELECT * FROM Option WHERE id=".$id)->fetch();

		$place = $option['place'];
	
		$count = $db->query("SELECT COUNT(*) FROM Option WHERE question_id='".$option['question_id']."'")->fetchColumn();

		if ( ($updown=="up") && ($place > 1) ){

			$option_to_push_down = $db->query("SELECT * FROM Option WHERE question_id='".$option['question_id']."' AND place='".($place-1)."'")->fetch();
			
			$db->query("UPDATE Option SET place = ".($option_to_push_down['place']+1)." WHERE id='".$option_to_push_down['id']."'");
			$db->query("UPDATE Option SET place = ".($option['place']-1)." WHERE id='".$option['id']."'");

		}
		elseif ( ($updown=="down") && ($place < $count) ){

			$option_to_push_up = $db->query("SELECT * FROM Option WHERE question_id='".$option['question_id']."' AND place='".($place+1)."'")->fetch();
			$db->query("UPDATE Option SET place = ".($option_to_push_up['place']-1)." WHERE id='".$option_to_push_up['id']."'");
			$db->query("UPDATE Option SET place = ".($option['place']+1)." WHERE id='".$option['id']."'");	

		}
	
	
		returnBack($option['question_id']);

	break;


default:
  echo "
<style>
.popover {
	max-width:700px;
	width: auto;
	height: auto;
#	max-height:200px;
}

</style>";

  echo "<div style='position:fixed; bottom:100px; right:50px'><a href='#'><u>Go to top</u></a></div>";

$questions = $db->query('SELECT * FROM Question ORDER BY place ASC')->fetchAll();

  echo "<center><br/><h1><a href='index.php'>Assessment of space needs at the LPC in 2013</a>: Editor</h1></br></center><hr/>";

  echo "<div class='container-fluid'>";
  echo "  <div class='row-fluid'>";
  echo "    <div class='span3'>";
  echo "       <div id='navbar'>";
  echo "	<center><h3>Questions</h3><hr/></center>";

  echo "          <ul class='nav'>";
	
  $question_index = 1;	
  $separator_index = 1;
	foreach($questions as $question)
    {   
	$question_index_str = "";
	$separator_style = " separator_left";	
	if ($question['is_separator']==0){
		$question_index_str = "Q ".$question_index.". ";
		$separator_style = "";	
		$question_index++;
	}
	else{

		$question_index_str = "Block ".$separator_index.". ";		
		$separator_index++;
	}	

	echo "<li class='rounded ".$separator_style."'><a href='#question".$question['id']."' title='".printSafe($question['title'])."'><b>".$question_index_str."</b>".printSafe(trim_text($question['title'], 60))."</a></li>";
        
	
    }
  echo "          </ul>";
  printAddQuestion();
  echo "       </div>";//navbar
  echo "    </div>";//span2
	

  echo "<div class='span8'>";  
    $question_index = 1;
    $separator_index = 1;	
    foreach($questions as $question)
    {

	$question_index_str = "";
	$separator_style = " separator_right";	
	if ($question['is_separator']==0){
		$question_index_str = "Question ".$question_index.". ";
		$separator_style = "";	
		$question_index++;
	}
	else{
		$question_index_str = "Block ".$separator_index.". ";		
		$separator_index++;
	}

	echo "<div class='question rounded' id='question".$question['id']."'>";
	echo "<div class='rounded questionHead ".$separator_style."'>";
	printEditQuestion($question);
	echo "<div class='title'><b>".$question_index_str."</b>".printSafe($question['title'])." ".questionInfo($question)."</div>";
	echo "<div style='clear:both'></div>";
	echo "</div>";
        if ($question['is_separator'] ==0){
		$options = $db->query('SELECT * FROM Option WHERE question_id='.$question['id'].' ORDER BY place ASC');	

		foreach($options as $option){			
        		printEditOption($option);
			printOption($option);				

			echo "<div style='clear:both'></div>";
	        }
	
		printAddOption($question); 
	}

	echo "</div>"; // question


    }
echo "</div>"; // span10
echo "</div>"; // row-fluid
echo "</div>"; // container-fluid

}



    // close the database connection
    $db = NULL;
  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }
echo "<div style='height:100px'></div>";
include_once("footer.php");
?>
#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"


#include "TGFrame.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TG3DLine.h"
#include "TGLViewer.h"
#include "TGSlider.h"
#include "TGTab.h"
#include "TGTextView.h"
#include "TGTextEdit.h"
#include "TGNumberEntry.h"
#include "TGHtml.h"

FWLayoutBuilder::FWLayoutBuilder(TGCompositeFrame *window, bool expandY)
   : m_window(window),
     m_currentFrame(0),
     m_floatLeft(false),
     m_topSpacing(0),
     m_leftSpacing(0),
     m_currentHints(0),
     m_currentFrameHints(0)
{
   TGVerticalFrame *mainFrame = new TGVerticalFrame(window);
   TGLayoutHints *hints = new TGLayoutHints(expandY ? kLHintsExpandX|kLHintsExpandY : kLHintsExpandX, 
                                            0, 0, 0, 0);
   m_window->AddFrame(mainFrame, hints);
   m_framesStack.push_back(mainFrame);
}

FWLayoutBuilder &
FWLayoutBuilder::newRow(void)
{
   m_currentFrameHints = m_currentHints = new TGLayoutHints(kLHintsExpandX);
   m_currentFrame = new TGHorizontalFrame(m_framesStack.back());
   m_framesStack.back()->AddFrame(m_currentFrame, m_currentHints);
   return *this;
}


   
FWLayoutBuilder &
FWLayoutBuilder::indent(int left /*= 2*/, int right /* = -1*/)
{
   if (right < 0)
      right = left;
      
   TGVerticalFrame *parent = m_framesStack.back();
   TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX|kLHintsExpandY, 
                                            left, right, 0, 0);
   m_currentHints = 0;
   m_framesStack.push_back(new TGVerticalFrame(parent));
   parent->AddFrame(m_framesStack.back(), hints);
   return newRow().expand(true, false);
}

/** Return the last vertical frame, for more control on the layout. */
TGVerticalFrame *
FWLayoutBuilder::verticalFrame(void)
{
   assert(m_framesStack.size());
   return m_framesStack.back();
}

/** Removes all the frames on the stack since last indent. */
FWLayoutBuilder &
FWLayoutBuilder::unindent(void)
{
   assert(!m_framesStack.empty());
   m_framesStack.pop_back();
   return *this;
}

/** Make sure that the current layout element is going to float on the 
    left of the next one.
  */
FWLayoutBuilder &
FWLayoutBuilder::floatLeft(size_t spacing)
{
   m_floatLeft = true;
   m_leftSpacing = spacing;
   return *this;
}

FWLayoutBuilder &
FWLayoutBuilder::spaceUp(size_t spacing)
{
   if (m_currentHints)
      m_currentHints->SetPadTop(spacing);
   return *this;
}

FWLayoutBuilder &
FWLayoutBuilder::spaceDown(size_t spacing)
{
   if (m_currentHints)
      m_currentHints->SetPadBottom(spacing);
   return *this;
}

FWLayoutBuilder &
FWLayoutBuilder::spaceLeft(size_t spacing)
{
   if (m_currentHints)
      m_currentHints->SetPadLeft(spacing);
   return *this;
}

FWLayoutBuilder &
FWLayoutBuilder::spaceRight(size_t spacing)
{
   if (m_currentHints)
      m_currentHints->SetPadRight(spacing);
   return *this;
}


FWLayoutBuilder &
FWLayoutBuilder::frameSpaceUp(size_t spacing)
{
   if (m_currentFrameHints)
      m_currentFrameHints->SetPadTop(spacing);
   return *this;
}

FWLayoutBuilder &
FWLayoutBuilder::frameSpaceDown(size_t spacing)
{
   if (m_currentFrameHints)
      m_currentFrameHints->SetPadBottom(spacing);
   return *this;
}

FWLayoutBuilder &
FWLayoutBuilder::frameSpaceLeft(size_t spacing)
{
   if (m_currentFrameHints)
      m_currentFrameHints->SetPadLeft(spacing);
   return *this;
}

FWLayoutBuilder &
FWLayoutBuilder::frameSpaceRight(size_t spacing)
{
   if (m_currentFrameHints)
      m_currentFrameHints->SetPadRight(spacing);
   return *this;
}

/** Set whether or not the previous layout element should expand and
    in which direction.
  */
FWLayoutBuilder &
FWLayoutBuilder::expand(bool expandX /*= true*/, bool expandY /*= false*/)
{
   UInt_t style = 0;
   style |= expandX ? kLHintsExpandX : 0;
   style |= expandY ? kLHintsExpandY : 0;
   
   if (m_currentHints)
      m_currentHints->SetLayoutHints(style);
   return *this;
}

// Returns the next layout to be used.
TGLayoutHints *
FWLayoutBuilder::nextHints()
{
   if (m_floatLeft)
   {
      size_t left = m_leftSpacing;
      m_floatLeft = false;
      m_leftSpacing = 0;
      assert(m_currentHints);
      m_currentHints = new TGLayoutHints(kLHintsExpandX, left, 0, 
                                         m_currentHints->GetPadTop(), 0);
   }
   else
   {
      size_t top = m_topSpacing;
      m_topSpacing = 3;
      m_currentHints = new TGLayoutHints(kLHintsExpandX, 0, 0, top, 0);
   }
   return m_currentHints;
}

TGCompositeFrame *
FWLayoutBuilder::nextFrame()
{
   if (!isFloatingLeft())
      newRow();

   return currentFrame();
}

void
FWLayoutBuilder::frameForTab()
{
   m_currentFrame = new TGVerticalFrame(m_framesStack.back());
   m_currentFrame->SetBackgroundColor(0x000000);
   m_framesStack.back()->AddFrame(m_currentFrame, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY));
   // m_frameStack.push_back(m_currentFrame);
}



/** Helper class to construct dialogs in a more readable ways.

    Encapsulated TGUI layout hiccups and exposes the developer an API which
    allows to layout items in a top->bottom, right->left manner.
    
    Example:
    
      FWDialogBuilder builder(parent);
      parent.newRow(2)              // New row which has a 2 pixel padding on top.
            .addLabel("MyLabel:")    // A new label.
            .indent(20)             // Whatever follows is indented 20 pixels 
                                    // on the right.
            .addLabel("MyLabel2")   // Another label.
            .spaceDown(4)
            .addTextButton("Aligned to MyLabel2 ").floatLeft()
            .addTextButton("Same Row as previous")
            .unindent()              // back one level in the indentation.
            .addLabel("Aligned to MyLabel:")
            
    Because in ROOT layout and parenting of widgets are mixed we need to take
    responsibility for creating the widget objects (sigh!), so we have one
    "addXYZ" method per widget that can be added. If we find our way around
    this it would be better to have a generic "addWidget()" method and create
    widgets outside this class.
    
    TODO: For higher configurability we should have an 
          "addWithCallback(Callbak)"  method which can be used to specify a 
          generic widget creation action.
  */
FWDialogBuilder::FWDialogBuilder(TGCompositeFrame *window, 
                                 FWDialogBuilder *parent /*= 0*/, bool expandY)
   : FWLayoutBuilder(window, expandY),
     m_parent(parent),
     m_tabs(0)
{}

FWDialogBuilder &
FWDialogBuilder::newRow()
{
   FWLayoutBuilder::newRow();
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::indent(int left /* = 2*/, int right /* = -1*/)
{
   FWLayoutBuilder::indent(left, right);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::unindent(void)
{
   FWLayoutBuilder::unindent();
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::addLabel(const char *text, size_t fontSize /*= 12*/,
                          size_t weight /*= 0*/, TGLabel **out /*= 0*/)
{
   TGLabel *label = new TGLabel(nextFrame(), text);

   if (fontSize != 0)
   {
      FontStruct_t defaultFontStruct = label->GetDefaultFontStruct();
      try
      {
         TGFontPool *pool = gClient->GetFontPool();
         TGFont* defaultFont = pool->GetFont(defaultFontStruct);
         FontAttributes_t attributes = defaultFont->GetFontAttributes();
         label->SetTextFont(pool->GetFont(attributes.fFamily, fontSize, 
                                          attributes.fWeight, attributes.fSlant));
      } 
      catch(...)
      {
         // Ignore exceptions.
      }
   }

   label->SetTextJustify(kTextLeft);
   
   TGLayoutHints *hints = nextHints();
   UInt_t style = hints->GetLayoutHints() | kLHintsCenterY;
   hints->SetLayoutHints(style);
   currentFrame()->AddFrame(label, hints);
   
   return extract(label, out);
}

FWDialogBuilder &
FWDialogBuilder::addTextView(const char *defaultText /*= 0*/, 
                             TGTextView **out /*= 0*/)
{
   TGTextView *view = new TGTextView(nextFrame(), 100, 100);
   if (defaultText)
      view->AddLine(defaultText);
   currentFrame()->AddFrame(view, nextHints());
   expand(true, true);
   return extract(view, out);
}

FWDialogBuilder &
FWDialogBuilder::addHtml(TGHtml **out /*= 0*/)
{
  TGHtml *html = new TGHtml(nextFrame(), 100, 100);
  currentFrame()->AddFrame(html, nextHints());   
  expand(true, true);
  return extract(html, out);
}

FWDialogBuilder &
FWDialogBuilder::addTextEdit(const char *defaultText /*= 0*/,
                             TGTextEdit **out /*= 0 */)
{
   TGTextEdit *edit = new TGTextEdit(nextFrame(), 100, 100);
   if (defaultText)
      edit->AddLine(defaultText);
   currentFrame()->AddFrame(edit, nextHints());
   expand(true, true);
   return extract(edit, out);
}

FWDialogBuilder &
FWDialogBuilder::addColorPicker(const FWColorManager *manager,
                                FWColorSelect **out /*= 0*/)
{
   const char* graphicsLabel = " ";
   FWColorSelect *widget = new FWColorSelect(nextFrame(), graphicsLabel,
                                             0, manager, -1);
   
   currentFrame()->AddFrame(widget, nextHints());
   widget->SetEnabled(kFALSE);
   
   return extract(widget, out);
}

FWDialogBuilder &
FWDialogBuilder::addHSlider(size_t size, TGHSlider **out /*= 0*/)
{
   TGHSlider *slider = new TGHSlider(nextFrame(), size, kSlider1);
   currentFrame()->AddFrame(slider, nextHints());
   slider->SetRange(0, 100);
   slider->SetPosition(100);
   slider->SetEnabled(false);
   
   return extract(slider, out);
}

FWDialogBuilder &
FWDialogBuilder::addTextButton(const char *text, TGTextButton **out /*= 0*/)
{
   TGTextButton *button = new TGTextButton(nextFrame(), text);
   currentFrame()->AddFrame(button, nextHints());
   button->SetEnabled(false);
   
   return extract(button, out);
}

FWDialogBuilder &
FWDialogBuilder::addValidatingTextEntry(const char *defaultText, 
                                        FWGUIValidatingTextEntry **out /*= 0*/)
{
   FWGUIValidatingTextEntry *entry = new FWGUIValidatingTextEntry(nextFrame());
   currentFrame()->AddFrame(entry, nextHints());
   
   return extract(entry, out);
}

FWDialogBuilder &
FWDialogBuilder::addTextEntry(const char *defaultText, 
                              TGTextEntry **out)
{
   TGTextEntry *entry = new TGTextEntry(nextFrame());
   currentFrame()->AddFrame(entry, nextHints());
   entry->SetEnabled(kFALSE);
   
   return extract(entry, out);
}

FWDialogBuilder &
FWDialogBuilder::addNumberEntry(float defaultValue, size_t digits,
                                TGNumberFormat::EStyle style,
                                int min, int max,
                                TGNumberEntry **out /*= 0*/)
{
   TGNumberEntry *entry = new TGNumberEntry(nextFrame(), defaultValue, 
                                            digits, -1, style,
                                            TGNumberFormat::kNEAAnyNumber,
                                            TGNumberFormat::kNELLimitMinMax,
                                            min, max);
   currentFrame()->AddFrame(entry, nextHints());
   entry->GetNumberEntry()->SetEnabled(kFALSE);
   entry->GetButtonUp()->SetEnabled(kFALSE);
   entry->GetButtonDown()->SetEnabled(kFALSE);
   return extract(entry, out);
}

FWDialogBuilder &
FWDialogBuilder::addCheckbox(const char *text, TGCheckButton **out /*= 0*/)
{
   TGCheckButton *button = new TGCheckButton(nextFrame(), text);
   button->SetState(kButtonDown, false);
   button->SetEnabled(false);
   currentFrame()->AddFrame(button, nextHints());
   
   return extract(button, out);
}

FWDialogBuilder &
FWDialogBuilder::addTable(FWTableManagerBase *manager, 
                          FWTableWidget **out /*= 0*/)
{
   expand(true, true);
   TGCompositeFrame *frame = verticalFrame();
   TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX|kLHintsExpandY);
   FWTableWidget *table = new FWTableWidget(manager, frame);
   frame->AddFrame(table, hints);
   return extract(table, out);
}

FWDialogBuilder &
FWDialogBuilder::addHSeparator(size_t horizontalPadding /*= 4*/, 
                               size_t verticalPadding /*= 3*/)
{
   TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX,
                                            horizontalPadding,
                                            horizontalPadding,
                                            verticalPadding,
                                            verticalPadding);

   TGHorizontal3DLine* separator = new TGHorizontal3DLine(nextFrame(), 200, 2);
   currentFrame()->AddFrame(separator, hints);
   return newRow();
}

/** Support for tabs.
 
    This is done by creating a new DialogBuilder and returning it for each
    of the added tabs.
    
    builder.tabs()               // Adds a TGTab widget to the current frame.
           .beginTab("Foo")      // Add a tab to the TGTab.
           .textButton("In Foo") // This is inside the tab "Foo", the layouting
                                 // is independent from previous calls
                                 // since a separate builder was returned by
                                 // 
           .endTab("Foo")        // End of the tab.
           .beginTab("Bar")
           .endTab("")
           .untabs();            // Tabs completed.
           .textButton("Main scope") // This is on the same level as the TGTab.
    
  */
FWDialogBuilder &
FWDialogBuilder::tabs(TGTab **out)
{
 
   // m_currentFrame = new TGVerticalFrame(m_framesStack.back());
   //m_framesStack.back()->AddFrame(m_currentFrame, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY));
   frameForTab();

   m_tabs = new TGTab(currentFrame());
   currentFrame()->AddFrame(m_tabs, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY));
   //expand(true, true);
   return extract(m_tabs, out);
}

FWDialogBuilder &
FWDialogBuilder::untabs(void)
{
   // No untabs() without tabs().
   assert(m_tabs);
   m_tabs = 0;
   return *this;
}

/** Adds a new tab called @a label.
    A new tab gets a new builder so that tab building is completely scoped.
  */
FWDialogBuilder &
FWDialogBuilder::beginTab(const char *label)
{
   TGCompositeFrame *tab = m_tabs->AddTab(label);
   
   FWDialogBuilder *builder = new FWDialogBuilder(tab, this, false);
   return builder->newRow();
}

/** When we are done with the tab, we delete ourself and return the parent.
  */
FWDialogBuilder &
FWDialogBuilder::endTab(void)
{
   FWDialogBuilder *parent = m_parent;
   delete this;
   return *parent;
}

FWDialogBuilder &
FWDialogBuilder::floatLeft(size_t spacing /*= 3*/)
{
   FWLayoutBuilder::floatLeft(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::spaceUp(size_t spacing /*= 3*/)
{
   FWLayoutBuilder::spaceUp(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::spaceDown(size_t spacing /*= 3*/)
{
   FWLayoutBuilder::spaceDown(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::spaceUpDown(size_t spacing /*= 3*/)
{
   FWLayoutBuilder::spaceUp(spacing);
   FWLayoutBuilder::spaceDown(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::spaceLeft(size_t spacing)
{
   FWLayoutBuilder::spaceLeft(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::spaceRight(size_t spacing)
{
   FWLayoutBuilder::spaceRight(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::spaceLeftRight(size_t spacing)
{
   FWLayoutBuilder::spaceLeft(spacing);
   FWLayoutBuilder::spaceRight(spacing);
   return *this;
}


// Frame spacing

FWDialogBuilder &
FWDialogBuilder::frameSpaceUp(size_t spacing /*= 3*/)
{
   FWLayoutBuilder::frameSpaceUp(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::frameSpaceDown(size_t spacing /*= 3*/)
{
   FWLayoutBuilder::frameSpaceDown(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::frameSpaceUpDown(size_t spacing /*= 3*/)
{
   FWLayoutBuilder::frameSpaceUp(spacing);
   FWLayoutBuilder::frameSpaceDown(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::frameSpaceLeft(size_t spacing)
{
   FWLayoutBuilder::frameSpaceLeft(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::frameSpaceRight(size_t spacing)
{
   FWLayoutBuilder::frameSpaceRight(spacing);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::frameSpaceLeftRight(size_t spacing)
{
   FWLayoutBuilder::frameSpaceLeft(spacing);
   FWLayoutBuilder::frameSpaceRight(spacing);
   return *this;
}


FWDialogBuilder &
FWDialogBuilder::expand(size_t expandX /*= true*/, size_t expandY /*= false*/)
{
   FWLayoutBuilder::expand(expandX, expandY);
   return *this;
}

FWDialogBuilder &
FWDialogBuilder::vSpacer(size_t size /* = 0*/)
{
   newRow().expand(true, true);
   
   TGFrame *frame;
   if (size) 
      frame = new TGFrame(nextFrame(), 1, size);
   else
      frame = new TGFrame(nextFrame());
   
   currentFrame()->AddFrame(frame, nextHints());

   expand(true, true);
   
   return *this;
}


FWDialogBuilder &
FWDialogBuilder::hSpacer(size_t size /* = 0*/)
{
   TGFrame *frame;
   if (size) 
      frame = new TGFrame(nextFrame(), size, 1);
   else
      frame = new TGFrame(nextFrame());
   
   currentFrame()->AddFrame(frame, nextHints());
   
   if (!size)
      expand(true, false);
   else
      expand(false, false);
   
   return *this;
}
